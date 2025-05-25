from einops.einops import rearrange, repeat
from loguru import logger

import torch
import torch.nn as nn
from .gaussian_process import CosKernel, GP, GP_Decoder, RRB
from .roma_cls_to_flow import cls_to_flow_refine, TransformerDecoder
from .roma_layers import Block, MemEffAttention

from time import time
from pdb import set_trace as bb


"""
train option

1. architecture option: concat something else or not

2) resolution option
query: 7x7? 15x15? 5x5? 3x3?
reference: 15x15? 7x7?

3. what if reference coords out of boundary [-1,1]

4. loss function design

5. proj_layer yes/no

6. grid_scale 8 => 4 (more training)
"""

class GPMatching(nn.Module):
    """FineMatching with s2d paradigm
    NOTE: use a separate class for d2d (sprase/dense flow) ?
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._type = config['s2d']['type']
        
        self.obtain_offset_method = config['s2d']['obtain_offset_method']

        self.left_point_movement = config['left_point_movement_window_size']
        self.best_left_strategy = config['best_left_strategy']
                
        gp_dim = 128
         
        kernel_temperature = 0.2
        learn_temperature = False
        no_cov = True
        kernel = CosKernel
        only_attention = False
        basis = "fourier"
        self.gp = GP(
            kernel,
            T=kernel_temperature,
            learn_temperature=learn_temperature,
            only_attention=only_attention,
            gp_dim=gp_dim,
            basis=basis,
            no_cov=no_cov,
        )
        
        # self.query_gp_decoder = GP_Decoder(internal_dim=config["d_model"])        
        self.ref_gp_decoder = GP_Decoder(internal_dim=128, classify=True, classify_dim=7)
        
        """
        # do not use: this architecture leads to slow training or worse performance
        gp_dim = 128
        feat_dim = 128
        decoder_dim = gp_dim + feat_dim
        cls_to_coord_res = 15  # can be changed into 15? 
        
        self.ref_gp_decoder = TransformerDecoder(
            blocks=nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
            hidden_dim=decoder_dim, 
            out_dim=cls_to_coord_res**2 + 1,
            is_classifier=True,
            amp = True,
            pos_enc = False,)
        """

        self.sigmoid = nn.Sigmoid()
        
        self.proj_layer = RRB(128, 128)
        # self.
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)

    def forward(self, features_reference_crop, features_query_crop, s2d_features_reference, s2d_features_query, \
            query_points_coarse, reference_points_coarse, \
            data, scales_relative_reference=None, track_mask=None, query_movable_mask=None, return_all_ref_heatmap=False):
        """
        Args:
            features_reference_crop: B * n_track * WW * C or B * n_track * WW * C (single view matching scenario)
            features_query_crop: B * n_track * (n_view - 1) * WW * C
            query_points_coarse: B * N_track * 2
            reference_points_coarse: B * n_track * (n_view - 1) * 2
            scales_relative_reference: B * n_track * (n_view -1)
            mask: B * n_track * (n_view - 1)
            query_movable_mask: B * N_track
            query_movable_mask: B * N_track
            keypoint_relocalized_offset: B * N_track, use keypoint detector to relocalize feature tracks
            data (dict)
        Update:
            data (dict):{
                "query_points_refined": [B, n_track, 2]
                "reference_points_refined": [B, n_view-1, n_track, 2]
                'std': [B, n_view-1, n_track]}
        """
        self.device = features_reference_crop.device
        B, n_track, n_query, WW, C = features_query_crop.shape
        assert B==1
        self.W, self.WW, self.C = data['W'], WW, C

        # Re-format:
        features_reference_crop = rearrange(features_reference_crop, "b t w c -> (b t) w c")
        features_query_crop = rearrange(features_query_crop, 'b t n (h1 w1) c -> (b t n) (h1 w1) c', h1=self.W, w1=self.W) # [M, n_view-1, WW, C]
        track_mask = rearrange(track_mask, 'b t n -> (b t) n') if track_mask is not None else None
        query_movable_mask = rearrange(query_movable_mask, 'b t -> (b t)') if query_movable_mask is not None else None
        # option1) 15 x 15 => GP + Decoder 
        # option2) 7 x 15 => GP + Decoder
        # option3) 7 x 7 => GP + Decoder
        # other option: remove boundary on side


        # leveraging feature map: 
        # option 1) reference's cross attention featmap: 128 C
        # option 2) option 1) + concat + target query's cross attention featmap : 128
        s2d_features_reference = rearrange(s2d_features_reference, 'b t w c -> (b t) w c')

        select_size = 7 # determine the size for gp decoder / and proj_layer / recommend >= 5 (due to RRB receptive size=5)
        features_query_selected = self.select_left_point(features_query_crop, data, select_size=select_size) # [M, C] or [M, n_view-1, C] (single view matching scenario)
        s2d_features_reference_selected = self.select_left_point(s2d_features_reference, data, select_size=select_size)        
        features_query_selected = rearrange(features_query_selected, 'b (h1 w1) c  -> b c h1 w1', h1=select_size, w1=select_size)        
        s2d_features_reference_selected = rearrange(s2d_features_reference_selected, 'b (h1 w1) c -> b c h1 w1', h1=select_size, w1=select_size)
        
        s2d_features_query = rearrange(s2d_features_query, 'b t n (h1 w1) c -> (b t n) c h1 w1', h1=self.W, w1=self.W)
        
        
        # ref also add
        features_reference_selected = self.select_left_point(features_reference_crop, data, select_size=select_size) # [M, C] or [M, n_view-1, C] (single view matching scenario)
        features_reference_selected = rearrange(features_reference_selected, 'b (h1 w1) c  -> b c h1 w1', h1=select_size, w1=select_size)        
        features_reference_selected = repeat(features_reference_selected, 'b c h w -> (b q) c h w', q=n_query)
        
        
        gp_feature_reference = self.proj_layer(s2d_features_reference_selected) # (b t) c select_size select_size
        # gp_feature_reference = rearrange(gp_feature_reference, '(b q) c h w -> b q c h w', q=1)
        # gp_feature_reference = repeat(gp_feature_reference, 'b q c h w -> b (q r) c h w', r=)
        
        gp_feature_query = self.proj_layer(s2d_features_query) # (b t n) c h2 w2
        
        ref_to_query = self.gp(gp_feature_reference, gp_feature_query, n_track=n_track, n_query=n_query)  # (b t n) c h1 w1
        # query_to_ref = self.gp(gp_feature_query, gp_feature_reference)
        
        # features_reference_selected = repeat(features_reference_selected, 'b c h w -> (b q) c h w', q=n_query)
        
        ref_to_query_coord, ref_to_query_certainty = self.ref_gp_decoder(ref_to_query, features_reference_selected, features_query_selected)

        if self.ref_gp_decoder.classify:
            ref_to_query_coord = cls_to_flow_refine(ref_to_query_coord)
            ref_to_query_coord = rearrange(ref_to_query_coord, 'b h w o -> b o h w')
        
        # query_to_ref_coord, query_to_ref_certainty = self.query_gp_decoder(query_to_ref, features_query_crop)
        
        # option0) gp + ref feature 
        # option1) concat gp + ref feature + query_i's feature
        # option2) from query to ref: bidirectional verification + std minimize
        
        # if want to apply roma's case: utis/utils.py 's cls_to_flow_refine & matcher.py, roma_models.py, transformer/__init__.py 's coordinate_decoder
        
        # the other architectue
        # 1) similar to roma: change it into classifier 
        # 2) from scale's flow & refiner 
        
        
        ref_to_query_certainty = self.sigmoid(ref_to_query_certainty)  # (track n_view-1) 1 W W
        ref_to_query_certainty = rearrange(ref_to_query_certainty, '(t n) o h w -> t (n o) (h w)', t=n_track, n=n_query) # o is one => t query_view (h w)
        ref_to_query_coord = rearrange(ref_to_query_coord, '(t n) o h w -> t n (h w) o', t=n_track, n=n_query) # t query_view (h w) 2
        
        # query_to_ref_certainty = self.sigmoid(query_to_ref_certainty)  
        
        track_inlier_mask = None
        if self.left_point_movement is not None: # at test time
            # track_mask: (b t) n 
            # sum_ref_to_query_certainty = torch.sum(ref_to_query_certainty, dim=1) # t query_view (w,w) -> t (w w)            
            sum_ref_to_query_certainty = torch.sum(ref_to_query_certainty * track_mask.unsqueeze(-1), dim=1)            
            ref_to_query_certainty_val, ref_to_query_certainty_index = torch.max(sum_ref_to_query_certainty, dim=-1)  # t / t
            
            # option 1 : naive sum
            # option 2: has highest num for which conf > 0.5                        
            
            win_size = int(sum_ref_to_query_certainty.shape[-1]**(0.5))
            # ref_to_query_certainty = torch.where(query_movable_mask, ref_to_query_certainty, sum_ref_to_query_certainty.shape[-1]//2)
            # ref_to_query_certainty = ref_to_query_certainty.unsqueeze(0)
            ref_to_query_certainty_index = torch.where(query_movable_mask, ref_to_query_certainty_index, sum_ref_to_query_certainty.shape[-1]//2)

            
            left_offset_x = ref_to_query_certainty_index % win_size
            left_offset_y = ref_to_query_certainty_index // win_size
            left_offset = torch.stack([left_offset_x, left_offset_y], dim=-1) # t 2
            query_offset_norm = (left_offset / (win_size - 1)) * 2 - 1 # NOTE: in the left window size coordinate
            query_offset_norm = query_offset_norm.unsqueeze(0)
            #has to be [B, n_track, 2]
            
            # best_index = torch.where(movable_mask, best_index, L//2)    
            ref_to_query_coord = ref_to_query_coord.permute(0,2,1,3) # n_track win**2 n_view 2
            coords_normed = ref_to_query_coord[torch.arange(ref_to_query_coord.shape[0]),ref_to_query_certainty_index].unsqueeze(0) # ref_to_query_coord : t, w*w, query_view, 2 / ref_to_query_certainty_index: t
            #has to be [B, n_track, query_view, 2]
            # track_inlier_mask = (ref_to_query_certainty[torch.arange(ref_to_query_certainty_index.shape[0]), :, ref_to_query_certainty_index]).mean(dim=-1) > 0.9
            

                            
        else:
            L = ref_to_query_coord.shape[-2]
            query_offset_norm = None
            coords_normed = ref_to_query_coord[...,L//2,:].unsqueeze(0)
            #has to be [B, n_track, query_view, 2]
            ref_to_query_certainty = ref_to_query_certainty[...,L//2].unsqueeze(0) # B(1) t query_view
        
        if query_offset_norm is not None:
            query_points_refined = self.build_moved_query(query_offset_norm, query_points_coarse, data)
        else:
            query_points_refined = query_points_coarse

        referece_points_refined = self.build_mkpts(coords_normed, reference_points_coarse, data, scales_relative_reference) # B(1) t query_view 2
                    
        return query_points_refined, referece_points_refined.transpose(1,2), ref_to_query_certainty.transpose(1,2), None, track_inlier_mask
        # best_index = torch.where(query_movable_mask, best_index, L//2)
        
        """
        if self.left_point_movement is not None:
            query_offset_norm, coords_normed, std, heatmap = \
                self._obtain_left_normalized_offset(coords_normed, std, heatmap_all_ref, \
                    track_mask=track_mask, movable_mask=query_movable_mask, keypoint_relocalized_offset=keypoint_relocalized_offset)
        else:
            query_offset_norm = None
            heatmap = heatmap_all_ref
        """
            
        """
        # De-format:
        coords_normed = rearrange(coords_normed, '(b t) n c -> b t n c', b=B)
        query_offset_norm = rearrange(query_offset_norm, '(b t) c -> b t c', b=B) if query_offset_norm is not None else None
        """
        
        # data.update({'fine_local_heatmap_pred': heatmap})

        """
        if query_offset_norm is not None:
            query_points_refined = self.build_moved_query(query_offset_norm, query_points_coarse, data)
        else:
            query_points_refined = query_points_coarse
        # compute absolute kpt coords: B * n_track * n_view-1 * 2
        referece_points_refined = self.build_mkpts(coords_normed, reference_points_coarse, data, scales_relative_reference)

        if return_all_ref_heatmap:
            heatmap_all_ref = rearrange(heatmap_all_ref, '(b t) m n w0 w1 -> b t n m w0 w1', b=B) if heatmap_all_ref is not None else None
            return query_points_refined, referece_points_refined.transpose(1,2), std, heatmap_all_ref,  # B * n_view-1 * n_track * 2, B * n_view-1 * n_track

        else:
            return query_points_refined, referece_points_refined.transpose(1,2), std, None,  # B * n_view-1 * n_track * 2, B * n_view-1 * n_track
        """
    
    def select_left_point(self, feat_f0, data, select_size=None):
        L = feat_f0.shape[-2] # [M, WW, C] or [M, n_view-1, WW, C]
        W = int(L ** .5)
        assert L % 2 == 1
        
        left_point_movement = self.left_point_movement
        if select_size is not None:
            left_point_movement = select_size
            
        if left_point_movement is None:
            feat_f0_picked = feat_f0[..., L//2, :]
        else:
            # assert not self.training
            assert left_point_movement > 0 and left_point_movement % 2 == 1 and left_point_movement <= W
            if len(feat_f0.shape) == 3:
                feat_f0 = rearrange(feat_f0, 'm (h w) c -> m h w c', h=W)
            else:
                raise NotImplementedError
            
            feat_f0_picked = feat_f0[..., (W//2 - left_point_movement//2):(W//2 + left_point_movement//2 + 1), (W//2 - left_point_movement//2):(W//2 + left_point_movement//2 + 1), :]
            feat_f0_picked = feat_f0_picked.flatten(-3, -2)

        return feat_f0_picked
        
    
    """
        # best_index[~movable_mask] = L // 2
        best_index = torch.where(movable_mask, best_index, L//2)
        left_offset_x = best_index % left_mv_win_size
        left_offset_y = best_index // left_mv_win_size
        left_offset = torch.stack([left_offset_x, left_offset_y], dim=-1)
        left_offset_norm = (left_offset / (left_mv_win_size - 1)) * 2 - 1 # NOTE: in the left window size coordinate

        # Select offset corresponding track:
        m_ids = torch.arange(coord_normed_tentative.shape[0], device=coord_normed_tentative.device)
        coord_normed_selected, std_selected, heatmap_selected = map(lambda x: x[m_ids, best_index], [coord_normed_tentative, std, heatmap])

        return left_offset_norm, coord_normed_selected, std_selected, heatmap_selected
    """
    
    """
    def _obtain_normalized_offset(self, heatmap):
        # Args:
        #     heatmap: B * L * N * W * W or B * N * W * W
        
        std = None
        M, N, W = heatmap.shape[:3]
        obtain_offset_method = self.obtain_offset_method
        if obtain_offset_method == 'argsoftmax':
            coords_normalized, std = argsoftmax(heatmap)
        else:
            raise NotImplementedError
        return coords_normalized, std
    """
        
    def build_moved_query(self, coords_normed, query_points_coarse, data):
        """
        Args:
            coords_normed: B * n_tracks * 2
            query_points_coarse: B * n_tracks * 2 (in original scale)
        Return:
            query_points_refined: B * n_tracks * 2
        """
        W, scales_origin_to_fine = self.left_point_movement, data['scales_origin_to_fine_query']
        window_size = (W // 2)
        query_points_refined = query_points_coarse + (coords_normed * window_size * scales_origin_to_fine)
        return query_points_refined
    
    def build_mkpts(self, coords_normed, reference_points_coarse, data, scales_relative_reference=None):
        """
        Args:
            coords_normed: B * n_tracks * n_view-1 * 2
            reference_points_coarse: B * n_tracks * n_view-1 * 2 (in original scale)
            scales_relative: B * n_track * n_view-1, only available when scale align is enabled
        Return:
            reference_points_refined: B * n_tracks * n_view-1 * 2
        """
        # scale_origin_to_fine: B * n_view * (n_track-1) * 2
        W, WW, C, scales_origin_to_fine = self.W, self.WW, self.C, data['scales_origin_to_fine_reference']
        
        # mkpts1_f
        if scales_relative_reference is not None:
            window_size = (W // 2) * scales_relative_reference[..., None]
        else:
            window_size = (W // 2)
        referece_points_refined = reference_points_coarse + (coords_normed * window_size * scales_origin_to_fine.transpose(1,2))
        return referece_points_refined

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)
