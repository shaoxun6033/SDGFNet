import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.WaveletDecomposition import WaveletDecomposition

# --- Low-Level Graph Operation ---
class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()
    
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)
    def forward(self,x):
        return self.mlp(x)
    
class static_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(static_mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class dynamic_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dynamic_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)
    def forward(self,x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)
        adj_final = (adj0 + adj1) / 2
        
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj_final)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)

        return ho1
   
# --- Model Branches ---
class GraphBranch(nn.Module):
    def __init__(self, configs):
        super(GraphBranch, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.num_nodes = configs.enc_in
        self.dropout = configs.dropout
        self.gcn_depth = configs.gcn_depth
        self.propalpha = configs.propalpha
        self.conv_channel = configs.conv_channel

        # self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.start_conv = nn.Conv2d(1, self.conv_channel, kernel_size=(1, 1))
        self.gcn = static_mixprop(self.conv_channel, self.conv_channel, self.gcn_depth, self.dropout, self.propalpha)
        self.end_conv = nn.Conv2d(self.conv_channel, self.seq_len, kernel_size=(1, self.seq_len))
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(self.seq_len)
        self.eps = 1e-9
        
    def compute_rbf_kernel_similarity(self, x, sigma=1.0):
        # x shape: [B, L, N]
        B, L, N = x.shape
        
        # Average the time series over the batch dimension first
        x_mean = torch.mean(x, dim=0) # -> [L, N]
        
        # Transpose to get [Nodes, Length] for distance calculation
        x_mean = x_mean.transpose(0, 1) # -> [N, L]

        # 1. Calculate the squared Euclidean distance between all pairs of nodes.
        # A fast way to do this is using the formula: ||a-b||^2 = ||a||^2 - 2a^T b + ||b||^2
        a_sq = torch.sum(x_mean**2, dim=1, keepdim=True) # -> [N, 1]
        b_sq = a_sq.transpose(0, 1)                      # -> [1, N]
        dot_product = torch.matmul(x_mean, x_mean.transpose(0, 1)) # -> [N, N]
        
        # The squared distance matrix
        dist_sq = a_sq - 2 * dot_product + b_sq # -> [N, N]
        
        # 2. Apply the Gaussian (RBF) kernel formula.
        # The adjacency is exp(-dist^2 / (2 * sigma^2))
        adj_matrix = torch.exp(-dist_sq / (2 * sigma**2))
        
        return adj_matrix
    def compute_pcc(self, x):
        """Computes Pearson Correlation Coefficient matrix."""
        # x_centered = x - x.mean(dim=1, keepdim=True)
        # cov = torch.einsum('bln,blm->bnm', x_centered, x_centered)
        # sum_sq_centered = torch.einsum('bln,bln->bn', x_centered, x_centered)
        # std_dev = torch.sqrt(sum_sq_centered + self.eps)
        # adj = cov / (torch.einsum('bn,bm->bnm', std_dev, std_dev) + self.eps)
        # adj = torch.mean(adj, dim=0)
        # # adj = F.relu(adj)
        # adj = adj ** 2
        
        adj = F.softmax(adj, dim=1)
        return torch.nan_to_num(adj)

    def forward(self, x, adj=None):
        B, N, T = x.shape
        
        if adj is None:
            adj = self.compute_pcc(x)
        # --- Residual Path ---
        # Project input to [B, Nodes, d_model] for the residual connection
        # residual = self.input_proj(x.permute(0, 2, 1))
        residual = x.permute(0, 2, 1)
        # --- Main Path ---
        # Use the projected residual as the input for the main path
        # [B, Nodes, d_model] -> [B, d_model, Nodes] for Conv1d
        out = residual.unsqueeze(1)                     # [B, 1, N, d_model]
        out = self.start_conv(out)                      # [B, conv_channel, N, d_model]
        out = self.gcn(out, adj)                        # [B, conv_channel, N, d_model]
        out = self.gelu(out)                            # [B, conv_channel, N, d_model]
        out = self.end_conv(out).squeeze().permute(0, 2, 1)             # [B, 1, N, d_model] -> [B, N, d_model]

        # --- Fusion ---
        # Add residual connection
        # Permute 'out' back to [B, Nodes, d_model] to match 'residual'
        final_out = self.norm(residual + out).permute(0, 2, 1)
        
        # Return in the expected shape for the fusion layer: [B, d_model, Nodes]
        return final_out


class WaveletBranch(nn.Module):
    """
    Dynamic Graph Branch via Wavelet Decomposition.
    Generates and sparsifies a dynamic graph representation for each frequency component.
    """
    def __init__(self, configs):
        super(WaveletBranch, self).__init__()
        self.d_model = configs.d_model
        self.num_nodes = configs.enc_in
        self.seq_len = configs.seq_len
        self.decomp_level = configs.decomp_level
        self.conv_channel = configs.conv_channel
        self.dropout = configs.dropout
        self.gcn_depth = configs.gcn_depth
        self.propalpha = configs.propalpha
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.wavelet_decomp = WaveletDecomposition(wavelet='db4', level=self.decomp_level, device=self.device)
        self.start_conv = nn.Conv2d(1, self.conv_channel, kernel_size=(1, 1))      
        self.dy_gcn = dynamic_mixprop(self.conv_channel, self.conv_channel, self.gcn_depth, self.dropout, self.propalpha)
        self.end_conv = nn.Conv2d(self.conv_channel, self.seq_len, kernel_size=(1, self.seq_len))
        
        self.k = int(0.6)

    def forward(self, x):
        x_decomp = self.wavelet_decomp(x)
        dynamic_graphs_out = []
        for i, x_part in enumerate(x_decomp):
            x_part_permuted = x_part.permute(0, 2, 1)
            # x_part_permuted = self.input_proj(x_part_permuted)
            x_part_permuted = x_part_permuted.unsqueeze(1)
            x_conv = self.start_conv(x_part_permuted)
            x_dy_gcn = self.dy_gcn(x_conv)
            x_dy_gcn = self.end_conv(x_dy_gcn)
            if self.k > 0:
                threshold = torch.topk(x_dy_gcn, self.k, dim=2, largest=True).values[..., -1, :].unsqueeze(2)
                mask = (x_dy_gcn >= threshold).float()
                x_dy_gcn_sparse = x_dy_gcn * mask
            else:
                x_dy_gcn_sparse = x_dy_gcn

            graph_representation = F.adaptive_avg_pool2d(x_dy_gcn_sparse, (self.num_nodes, 1)).squeeze(-1)
            dynamic_graphs_out.append(graph_representation)
            
        return dynamic_graphs_out

class WaveletBranchSimple(nn.Module):
    """
    Ablation version:
    Only performs Wavelet Decomposition without graph convolution or sparsification.
    """
    def __init__(self, configs):
        super(WaveletBranchSimple, self).__init__()
        self.decomp_level = configs.decomp_level
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wavelet_decomp = WaveletDecomposition(
            wavelet='db4', 
            level=self.decomp_level, 
            device=self.device
        )

    def forward(self, x):
        x_decomp = self.wavelet_decomp(x)  # list of [B, L, N]
        return x_decomp
    
# --- Fusion and Post-Processing Blocks ---

class AttentionGatedFusion(nn.Module):
    """
    Fuses a static graph and multiple dynamic graphs using a Query-Key attention mechanism.
    """
    def __init__(self, seq_len, num_nodes, num_graphs, d_model=128):
        super(AttentionGatedFusion, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # 1. A projection layer to create a feature vector for each graph
        self.graph_feature_proj = nn.Linear(seq_len * num_nodes, d_model)
        
        # 2. Query-Key Attention components
        # A learnable "fusion context" vector
        self.query_vector = nn.Parameter(torch.randn(1, d_model))
        # A linear layer to project graph features into "Keys"
        self.key_projection = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    def forward(self, static_graph, dynamic_graphs):
        B, D, N = static_graph.shape
        all_graphs = [static_graph] + dynamic_graphs
        stacked_graphs = torch.stack(all_graphs, dim=1)
        K = stacked_graphs.size(1)  # Total number of graphs (1 static + dynamic)
        graph_flat_features = stacked_graphs.view(B, K, -1) # -> [B, K, L*N]
        
        # Project to a fixed dimension d_model
        graph_vectors = self.graph_feature_proj(graph_flat_features) # -> [B, K, d_model]
        graph_vectors = self.norm(graph_vectors)

        # 2. Compute Attention Weights
        # Expand the learnable query_vector to the batch size
        query = self.query_vector.expand(B, -1).unsqueeze(1) # -> [B, 1, d_model]
        
        # Project graph vectors to Keys
        keys = self.key_projection(graph_vectors) # -> [B, K, d_model]
        
        # Calculate dot-product attention scores
        # (B, 1, d) @ (B, d, K) -> (B, 1, K)
        attention_scores = torch.bmm(query, keys.transpose(1, 2))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1) # -> [B, 1, K]
        
        # 3. Fuse the graphs using the calculated weights
        # Reshape weights for broadcasting: [B, 1, K] -> [B, K, 1, 1]
        weights_reshaped = attention_weights.transpose(1, 2).unsqueeze(-1)
        
        # Weighted sum of the original graph features
        # (B, K, L, N) * (B, K, 1, 1) -> sum over K -> [B, L, N]
        fused_graph = (stacked_graphs * weights_reshaped).sum(dim=1)
        
        return fused_graph

class SimpleGraphFusion(nn.Module):
    """
    Fuses a static graph and multiple dynamic graphs by simple summation (ablation version).
    """
    def __init__(self):
        super(SimpleGraphFusion, self).__init__()

    def forward(self, static_graph, dynamic_graphs):
        # all_graphs: list of [B, L, N]
        all_graphs = [static_graph] + dynamic_graphs
        # 简单逐元素相加
        fused_graph = torch.stack(all_graphs, dim=0).sum(dim=0)  # [B, L, N]
        return fused_graph
    
class InceptionBlock(nn.Module):
    """
    Inception block with dilated convolutions and a residual connection.
    """
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.d_model = 512
        self.conv1 = nn.Conv1d(in_channels, self.d_model, kernel_size=1)
        self.conv1x2 = nn.Conv1d(self.d_model, out_channels, kernel_size=3, padding='same', dilation=1)
        self.conv1x3 = nn.Conv1d(self.d_model, out_channels, kernel_size=3, padding='same', dilation=2)
        self.conv1x6 = nn.Conv1d(self.d_model, out_channels, kernel_size=5, padding='same', dilation=1)
        self.conv1x7 = nn.Conv1d(self.d_model, out_channels, kernel_size=5, padding='same', dilation=2)
        
        self.bottleneck = nn.Conv1d(out_channels * 4, out_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(self.d_model, out_channels, kernel_size=1)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        residual = self.residual_conv(x)
        
        c1 = self.conv1x2(x)
        c2 = self.conv1x3(x)
        c3 = self.conv1x6(x)
        c4 = self.conv1x7(x)
        
        x_cat = torch.cat([c1, c2, c3, c4], dim=1)
        x_out = self.bottleneck(x_cat)
        
        x_fused = x_out + residual
        
        # Permute for LayerNorm, apply it, and permute back
        return self.ln(x_fused.permute(0, 2, 1)).permute(0, 2, 1)
    
class MLPBlock(nn.Module):
    """
    MLP block with residual connection and LayerNorm (for ablation study).
    """
    def __init__(self, in_channels, out_channels, hidden_dim=512):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)   
        self.fc2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)  
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)  
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = self.residual_conv(x)  # shortcut
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + residual  
        return self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)