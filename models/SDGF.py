import torch
import torch.nn as nn
from utils.RevIN import RevIN
from layers.modules import GraphBranch, WaveletBranch, AttentionGatedFusion, InceptionBlock
from layers.modules import WaveletBranchSimple, SimpleGraphFusion
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # --- 1. Normalization Layer ---
        self.revin = RevIN(self.enc_in, affine=True)
        
        # --- 2. Model Branches ---
        self.graph_branch = GraphBranch(configs)
        self.wavelet_branch = WaveletBranch(configs)

        # self.wavelet_branch_simple = WaveletBranchSimple(configs)

        # --- 3. Fusion Layer ---
        num_dynamic_graphs = configs.decomp_level + 1
        self.fusion = AttentionGatedFusion(self.seq_len, self.enc_in, num_graphs=1 + num_dynamic_graphs)

        self.fusionsimple = SimpleGraphFusion()
        # --- 4. Post-Fusion Processing ---
        self.inception = InceptionBlock(configs.seq_len, configs.seq_len)
        
        # --- 5. Output Layer ---
        self.mlp = nn.Linear(configs.seq_len, configs.pred_len)
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, adj=None):
        """
        Forward pass of the SDGF model.
        Args:
            x_enc: Input tensor of shape [Batch, seq_len, num_nodes]
            adj: Optional predefined adjacency matrix.
        Returns:
            Output tensor of shape [Batch, pred_len, num_nodes]
        """
        # 1. Normalization
        x_normalized = self.revin(x_enc, 'norm') # Note: your RevIN used 'norm', I changed to 'normalize' for clarity

        # If no adjacency matrix is provided, compute it once
        shared_adj = adj
        if shared_adj is None:
            # shared_adj = self.graph_branch.compute_pcc(x_normalized)
            shared_adj = self.graph_branch.compute_rbf_kernel_similarity(x_normalized)


        # 2. Branch Processing
        out_graph_static = self.graph_branch(x_normalized, shared_adj)
        out_wavelet_dynamic_list = self.wavelet_branch(x_normalized)

        # 3. Fusion
        x_fused = self.fusion(out_graph_static, out_wavelet_dynamic_list)

        # 4. Post-Fusion Block
        x_inception = self.inception(x_fused)
        # x_inception = self.inceptionsimple(x_fused)

        # 5. KAN Output
        x_kan_in = x_inception.permute(0, 2, 1) 
        
        x_kan_out = self.mlp(x_kan_in)
        output = x_kan_out.permute(0, 2, 1)

        # 6. Denormalization
        final_output = self.revin(output, 'denorm') # Note: your RevIN used 'denorm'

        return final_output