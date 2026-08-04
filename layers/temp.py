
class vlinear(nn.Module):
    def __init__(self, num_vars, order, hidden_dim=128, device="cpu", options=None):#128 for SMD
        super().__init__()
        self.num_vars = num_vars  
        self.order = order*1  -1      
        self.device = device
        self.options = options or {}
        if "orth_transformer_multi_modality" in options and options["orth_transformer_multi_modality"]:
            self.multi_modal = True
            self.orth_transformer = OrthTransform_multi_modal(
                dataset_obj=options['dataset_obj'],
                save_path=options['save_path'],
                time_lag=options['time_lag'],
                device=device
            )
        else:
            self.multi_modal = False
            self.orth_transformer = options.get('orth_transformer') 
        
        # 1. Delta Biases (Faithful to Model logic)
        # These act as "Learned Context" for the orthogonal domain
        # delta1: [1, Channels, 1, Lag]
        self.delta_latent1 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))
        self.delta_latent2 = nn.Parameter(torch.randn(1, num_vars, hidden_dim))

        # Projection to match the output 'order'
        self.bias_proj = nn.Linear(hidden_dim, self.order)
        
        # 2. Updated Embeddings 
        # In the Model code, embeddings are often 1D and expanded
        #self.embeddings = nn.Parameter(torch.randn(1, hidden_dim))
        self.embeddings = nn.Parameter(torch.randn(1, num_vars, 1, hidden_dim))
        # 3. Projection matching the Model's logic
        #self.temporal_proj = nn.Linear(self.order, hidden_dim)
        self.temporal_proj = nn.Linear(1, hidden_dim)
        #self.mom = TinyMoM(
        #    hidden_dim,
        #    num_memories=4,
        #    top_k=2
        #)
        #self.use_MoM = options.get(
        #    "use_MoM",
        #    True
        #)
        #self.temporal_proj = nn.Sequential(
        #    nn.Linear(1, hidden_dim // 2),
        #    nn.GELU(),
        #    nn.Linear(hidden_dim // 2, hidden_dim)
        #)
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, self.order) 
        )
        #self.revin = RevIN(num_vars)
        self.use_temporal_mixer = options.get('temporal_mixer', False)
        if self.use_temporal_mixer:
            self.temporal_mixer = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim
            )
        

    def forward(self, inputs: torch.Tensor):
        B, O_curr, P = inputs.shape # [B, Window, Sensors] e.g., [B, 2, 51]
        
        # --- 1. Step into Orthogonal Domain ---
        if self.orth_transformer is None:
            x_orth = inputs.transpose(1, 2) # [B, 51, 2]
        else:
            if self.multi_modal:
                md, ld, td = self.md, self.ld, self.td

                xm = inputs[:, :, :md]
                xl = inputs[:, :, md:md+ld]
                xt = inputs[:, :, md+ld:md+ld+td]

                xm, xl, xt = self.orth_transformer(xm, xl, xt)

                assert xm.shape[1] == xl.shape[1] == xt.shape[1], "Orth mismatch in time axis"

                x_orth = torch.cat([xm, xl, xt], dim=1)
            else:
                x_orth = self.orth_transformer(inputs)
        
        # --- 2. Apply Delta1 Latent Bias ---
        # Project delta_latent1 [1, P, H] -> [1, P, Order]
        # Then unsqueeze to [1, P, 1, Order] for broadcasting
        d1 = self.bias_proj(self.delta_latent1).unsqueeze(-2) 
        x_orth_biased = x_orth.unsqueeze(-2) + d1 # [B, P, 1, Order]
        
        # --- 3. Truly Dynamic Latent Generation ---
        # [B, P, 1, Order] -> [B, Order, P, 1]
        x_t = x_orth_biased.squeeze(-2).transpose(1, 2).unsqueeze(-1)
        
        # Project each sensor at each time step into H-space
        # cond: [B, Order, P, H]
        cond = self.temporal_proj(x_t) * self.embeddings.transpose(1, 2) 
        # [B,T,P,H] -> [B,P,H,T]
        if self.use_temporal_mixer:
            cond_mix = cond.permute(0,2,3,1)
            H = cond_mix.shape[2]
            T = cond_mix.shape[3]
    #
            cond_mix = self.temporal_mixer(cond_mix.reshape(B*P, H, T))
            cond_mix = cond_mix.reshape(B, P, H, T).permute(0,3,1,2)

            cond = cond + cond_mix
        else:
            cond = cond
        # --- 4. Dynamic AERCA Coefficients ---
        # Creates a unique PxP matrix for every step in the window
        coeffs_time = torch.einsum('btph, btqh -> btpq', cond, cond)
        coeffs_time = torch.tanh(coeffs_time)

        # --- 5. Prediction (Forecasting) ---
        # Aggregate temporal info using max pooling (as per your best results)
        #if not self.use_MoM:
        z_final, _ = torch.max(cond, dim=1) # [B, P, H]
        #else:
        #    z_final = self.mom(cond).max(dim=1).values
        
        # Apply the second Latent Bias to the forecast
        # vf(z_final) -> [B, P, Order]
        # d2 -> [1, P, Order]
        d2 = self.bias_proj(self.delta_latent2)
        v_pred = self.vf(z_final) + d2 # [B, P, Order]

        if self.orth_transformer is None:
            preds_all_time = v_pred.transpose(1, 2) # [B, Order, P]
        else:
            preds_all_time = self.orth_transformer.inverse(v_pred)
        
        # Final forecast is the last step of the predicted window
        preds = preds_all_time[:, -1, :] 
        coeffs_freq = coeffs_time[:, 0, :, :] # First step coefficients

        return preds, coeffs_time, coeffs_freq
