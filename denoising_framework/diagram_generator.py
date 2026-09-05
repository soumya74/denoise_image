import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network_diagram(output_filename="disentangle_network_architecture.png"):
    fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, w, h, text, bg="#E8EEF5", border="#2B4C7E", fontsize=9, bold=True):
        weight = "bold" if bold else "normal"
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            linewidth=1.5, edgecolor=border, facecolor=bg
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1A1A1A", multialignment="center")
        return rect

    def arrow(x1, y1, x2, y2, color="#4A5568", style="->", lw=1.6, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=3, shrinkB=3))

    # Header Title
    ax.text(10.0, 10.4, "Dual-Decoder Disentanglement Network: 7-Loss Architecture Formulation",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")

    # Inputs
    box(0.5, 6.8, 1.8, 1.3, "Noisy Image\ny ~ (x + n)", bg="#FED7AA", border="#EA580C")
    box(0.5, 2.5, 1.8, 1.3, "Clean Image\nx (Ground Truth)", bg="#BBF7D0", border="#16A34A")

    # Shared Encoder
    box(3.2, 4.2, 2.0, 4.2, "Shared\nEncoder\nNetwork", bg="#E0E7FF", border="#4338CA", fontsize=11)
    arrow(2.3, 7.45, 3.2, 6.8, color="#EA580C")
    arrow(2.3, 3.15, 3.2, 4.8, color="#16A34A")

    # Latent Heads (Noisy Path)
    box(6.0, 7.8, 2.4, 1.0, "Content Head\nz_c^y  [28 channels]", bg="#BBF7D0", border="#16A34A")
    box(6.0, 6.0, 2.4, 1.0, "Noise Head\nz_n^y  [4 channels]", bg="#FED7AA", border="#EA580C")
    arrow(5.2, 7.1, 6.0, 8.3, color="#EA580C")
    arrow(5.2, 6.4, 6.0, 6.5, color="#EA580C")

    # Latent Heads (Clean Path)
    box(6.0, 3.4, 2.4, 1.0, "Content Head\nz_c^x  [28 channels]", bg="#BBF7D0", border="#16A34A")
    box(6.0, 1.6, 2.4, 1.0, "Noise Head\nz_n^x  [4 channels]", bg="#FED7AA", border="#EA580C")
    arrow(5.2, 5.2, 6.0, 3.9, color="#16A34A")
    arrow(5.2, 4.6, 6.0, 2.1, color="#16A34A")

    # Decoders
    box(9.6, 7.6, 2.2, 1.3, "Clean Decoder\n(D_clean)", bg="#C7D2FE", border="#3730A3")
    box(9.6, 5.7, 2.2, 1.3, "Noise Decoder\n(D_noise)", bg="#C7D2FE", border="#3730A3")
    box(9.6, 3.1, 2.2, 1.3, "Clean Decoder\n(D_clean)", bg="#C7D2FE", border="#3730A3")

    arrow(8.4, 8.3, 9.6, 8.25, color="#16A34A")
    arrow(8.4, 6.5, 9.6, 6.35, color="#EA580C")
    arrow(8.4, 3.9, 9.6, 3.75, color="#16A34A")

    # Decoded Outputs
    box(12.7, 7.8, 1.9, 0.9, "Reconstructed\nClean (x̂)", bg="#DCFCE7", border="#15803D")
    box(12.7, 5.9, 1.9, 0.9, "Reconstructed\nNoise (n̂)", bg="#FFEDD5", border="#C2410C")
    box(12.7, 3.3, 1.9, 0.9, "Self Clean\n(x̂_clean)", bg="#DCFCE7", border="#15803D")

    arrow(11.8, 8.25, 12.7, 8.25)
    arrow(11.8, 6.35, 12.7, 6.35)
    arrow(11.8, 3.75, 12.7, 3.75)

    # Reconstructed Noisy (x̂ + n̂)
    box(15.2, 7.0, 0.8, 0.8, "+", bg="#F1F5F9", border="#475569", fontsize=14)
    arrow(14.6, 8.1, 15.2, 7.5)
    arrow(14.6, 6.5, 15.2, 7.2)

    box(16.5, 6.8, 2.0, 1.1, "Reconstructed\nNoisy (ŷ)\n= x̂ + n̂", bg="#FEF3C7", border="#D97706")
    arrow(16.0, 7.4, 16.5, 7.4)

    # --- 7 Explicit Loss Objective Boxes (Ordered 1 to 7) ---

    # 1. Clean Rec
    box(15.5, 9.4, 3.9, 0.8, "1. L_clean-rec = ||x̂ - x||₁", bg="#FEF2F2", border="#DC2626")
    arrow(14.6, 8.45, 15.5, 9.6, color="#DC2626")

    # 2. Noisy Rec
    box(15.5, 5.5, 3.9, 0.8, "2. L_noisy-rec = ||ŷ - y||₁", bg="#FEF2F2", border="#DC2626")
    arrow(17.5, 6.8, 17.5, 6.3, color="#DC2626")

    # 3. Noise Rec
    box(15.5, 8.2, 3.9, 0.8, "3. L_noise-rec = ||n̂ - (y - x)||₁", bg="#FEF2F2", border="#DC2626")
    arrow(14.6, 6.35, 15.5, 8.4, color="#DC2626")

    # 4. Clean Self
    box(15.5, 2.7, 3.9, 0.8, "4. L_clean-self = ||x̂_clean - x||₁", bg="#FEF2F2", border="#DC2626")
    arrow(14.6, 3.6, 15.5, 3.1, color="#DC2626")

    # 5. Latent Content Consistency
    box(6.0, 4.9, 2.4, 0.6, "5. L_content-latent = ||z_c^y - z_c^x||₁", bg="#FEF2F2", border="#DC2626", fontsize=7.2)
    arrow(7.2, 7.8, 7.2, 5.5, color="#DC2626", style="<->")
    arrow(7.2, 4.9, 7.2, 4.4, color="#DC2626", style="<->")

    # 6. Clean Latent Noise Suppression (Noise Zero)
    box(9.2, 1.6, 3.6, 0.8, "6. L_noise-zero = ||z_n^x||₁ → 0", bg="#FEF2F2", border="#DC2626")
    arrow(8.4, 2.1, 9.2, 2.0, color="#DC2626")

    # 7. Clean Cross (Consistency between x̂ and x̂_clean)
    box(15.5, 4.0, 3.9, 0.8, "7. L_clean-cross = ||x̂ - x̂_clean||₁", bg="#EFF6FF", border="#2563EB")
    arrow(13.65, 7.8, 15.5, 4.5, color="#2563EB", ls="--")
    arrow(13.65, 4.2, 15.5, 4.3, color="#2563EB", ls="--")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"--> Architecture diagram saved successfully to: {output_filename}")

if __name__ == "__main__":
    draw_network_diagram()