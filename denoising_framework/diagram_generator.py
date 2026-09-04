import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network_diagram(output_filename="disentangle_network_architecture.png"):
    fig, ax = plt.subplots(figsize=(19, 10), dpi=300)
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10)
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

    def arrow(x1, y1, x2, y2, color="#4A5568", style="->", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3))

    # Title
    ax.text(9.5, 9.5, "Dual-Decoder Disentanglement Network Architecture & 6 Loss Formulations",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")

    # Inputs
    box(0.5, 6.2, 1.8, 1.2, "Noisy Image\ny ~ (x + n)", bg="#FED7AA", border="#EA580C")
    box(0.5, 2.2, 1.8, 1.2, "Clean Image\nx (Ground Truth)", bg="#BBF7D0", border="#16A34A")

    # Shared Encoder
    box(3.2, 3.8, 2.0, 3.8, "Shared\nEncoder\nNetwork", bg="#E0E7FF", border="#4338CA", fontsize=11)
    arrow(2.3, 6.8, 3.2, 6.2, color="#EA580C")
    arrow(2.3, 2.8, 3.2, 4.4, color="#16A34A")

    # Latent Heads (Noisy Path)
    box(6.0, 7.2, 2.4, 1.0, "Content Head\nz_c^y  [28 channels]", bg="#BBF7D0", border="#16A34A")
    box(6.0, 5.4, 2.4, 1.0, "Noise Head\nz_n^y  [4 channels]", bg="#FED7AA", border="#EA580C")
    arrow(5.2, 6.5, 6.0, 7.7, color="#EA580C")
    arrow(5.2, 5.8, 6.0, 5.9, color="#EA580C")

    # Latent Heads (Clean Path)
    box(6.0, 3.2, 2.4, 1.0, "Content Head\nz_c^x  [28 channels]", bg="#BBF7D0", border="#16A34A")
    box(6.0, 1.4, 2.4, 1.0, "Noise Head\nz_n^x  [4 channels]", bg="#FED7AA", border="#EA580C")
    arrow(5.2, 4.8, 6.0, 3.7, color="#16A34A")
    arrow(5.2, 4.2, 6.0, 1.9, color="#16A34A")

    # Decoders
    box(9.6, 7.0, 2.2, 1.3, "Clean Decoder\n(D_clean)", bg="#C7D2FE", border="#3730A3")
    box(9.6, 5.1, 2.2, 1.3, "Noise Decoder\n(D_noise)", bg="#C7D2FE", border="#3730A3")
    box(9.6, 2.9, 2.2, 1.3, "Clean Decoder\n(D_clean)", bg="#C7D2FE", border="#3730A3")

    arrow(8.4, 7.7, 9.6, 7.7, color="#16A34A")
    arrow(8.4, 5.9, 9.6, 5.7, color="#EA580C")
    arrow(8.4, 3.7, 9.6, 3.6, color="#16A34A")

    # Decoded Outputs
    box(12.7, 7.2, 1.8, 0.9, "Reconstructed\nClean (x_hat)", bg="#DCFCE7", border="#15803D")
    box(12.7, 5.3, 1.8, 0.9, "Reconstructed\nNoise (n_hat)", bg="#FFEDD5", border="#C2410C")
    box(12.7, 3.1, 1.8, 0.9, "Self Clean\n(x_hat_clean)", bg="#DCFCE7", border="#15803D")

    arrow(11.8, 7.7, 12.7, 7.7)
    arrow(11.8, 5.8, 12.7, 5.8)
    arrow(11.8, 3.6, 12.7, 3.6)

    # Reconstructed Noisy (x_hat + n_hat)
    box(15.2, 6.3, 0.8, 0.8, "+", bg="#F1F5F9", border="#475569", fontsize=14)
    arrow(14.5, 7.5, 15.2, 6.9)
    arrow(14.5, 6.0, 15.2, 6.5)

    box(16.5, 6.1, 2.0, 1.1, "Reconstructed\nNoisy (y_hat)\n= x_hat + n_hat", bg="#FEF3C7", border="#D97706")
    arrow(16.0, 6.7, 16.5, 6.7)

    # Loss Callout Boxes
    # 1. Clean Rec
    box(15.0, 8.2, 3.5, 0.8, "1. L_clean-rec = ||x_hat - x||_1", bg="#FEF2F2", border="#DC2626")
    arrow(14.5, 7.7, 15.0, 8.4, color="#DC2626")

    # 2. Noisy Rec
    box(15.0, 5.0, 3.5, 0.8, "2. L_noisy-rec = ||y_hat - y||_1", bg="#FEF2F2", border="#DC2626")
    arrow(17.5, 6.1, 17.5, 5.8, color="#DC2626")

    # 3. Noise Rec
    box(15.0, 3.9, 3.5, 0.8, "3. L_noise-rec = ||n_hat - n||_1", bg="#FEF2F2", border="#DC2626")
    arrow(14.5, 5.6, 15.0, 4.3, color="#DC2626")

    # 4. Clean Self
    box(15.0, 2.7, 3.5, 0.8, "4. L_clean-self = ||x_hat_clean - x||_1", bg="#FEF2F2", border="#DC2626")
    arrow(14.5, 3.4, 15.0, 3.1, color="#DC2626")

    # 5. Latent Consistency (Bridge between z_c^y and z_c^x)
    box(6.0, 4.5, 2.4, 0.6, "5. L_latent = ||z_c^y - z_c^x||_1", bg="#FEF2F2", border="#DC2626", fontsize=7.5)
    arrow(7.2, 7.2, 7.2, 5.1, color="#DC2626", style="<->")
    arrow(7.2, 4.5, 7.2, 4.2, color="#DC2626", style="<->")

    # 6. Noise Zero (Constraint on z_n^x)
    box(8.8, 1.4, 3.2, 0.8, "6. L_noise-zero = ||z_n^x||_1 -> 0", bg="#FEF2F2", border="#DC2626")
    arrow(8.4, 1.9, 8.8, 1.9, color="#DC2626")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"--> Architecture diagram saved successfully to: {output_filename}")

if __name__ == "__main__":
    draw_network_diagram()