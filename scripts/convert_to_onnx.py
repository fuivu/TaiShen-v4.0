#!/usr/bin/env python3
"""
Convert HuggingFace Stable Diffusion models to ONNX format for mobile inference.

Usage:
    python convert_to_onnx.py --model runwayml/stable-diffusion-v1-5 --output ./onnx_models/
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline

def convert_text_encoder(pipe, output_dir):
    """Export CLIP text encoder to ONNX"""
    print("[1/3] Converting Text Encoder...")
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Dummy input
    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.long)

    # Trace
    text_encoder.eval()
    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            dummy_input,
            os.path.join(output_dir, "text_encoder.onnx"),
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
            opset_version=14,
            do_constant_folding=True
        )
    print(f"  ✓ Saved to {output_dir}/text_encoder.onnx")

def convert_unet(pipe, output_dir):
    """Export UNet to ONNX with dynamic shapes"""
    print("[2/3] Converting UNet...")
    unet = pipe.unet

    # Dummy inputs
    batch_size = 1
    latent_channels = 4
    height, width = 64, 64  # 512/8
    dummy_latents = torch.randn(batch_size, latent_channels, height, width)
    dummy_timestep = torch.tensor([500], dtype=torch.long)
    dummy_text_emb = torch.randn(batch_size, 77, 768)

    unet.eval()
    with torch.no_grad():
        torch.onnx.export(
            unet,
            (dummy_latents, dummy_timestep, dummy_text_emb),
            os.path.join(output_dir, "unet.onnx"),
            input_names=["latent", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "latent": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"}
            },
            opset_version=14,
            do_constant_folding=True
        )
    print(f"  ✓ Saved to {output_dir}/unet.onnx")

def convert_vae_decoder(pipe, output_dir):
    """Export VAE decoder to ONNX"""
    print("[3/3] Converting VAE Decoder...")
    vae = pipe.vae

    batch_size = 1
    latent_channels = 4
    height, width = 64, 64
    dummy_latents = torch.randn(batch_size, latent_channels, height, width)

    vae.eval()
    with torch.no_grad():
        # Only export the decoder part
        class VaeDecoderOnly(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.decoder = vae.decoder
                self.post_quant_conv = vae.post_quant_conv
            def forward(self, x):
                x = self.post_quant_conv(x)
                x = self.decoder(x)
                return x

        decoder_only = VaeDecoderOnly(vae)
        torch.onnx.export(
            decoder_only,
            dummy_latents,
            os.path.join(output_dir, "vae_decoder.onnx"),
            input_names=["latents"],
            output_names=["image"],
            dynamic_axes={"latents": {0: "batch", 2: "height", 3: "width"}},
            opset_version=14,
            do_constant_folding=True
        )
    print(f"  ✓ Saved to {output_dir}/vae_decoder.onnx")

def main():
    parser = argparse.ArgumentParser(description="Convert SD to ONNX")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", default="./onnx_models", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading model: {args.model}")

    dtype = torch.float16 if args.fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to("cpu")

    convert_text_encoder(pipe, args.output)
    convert_unet(pipe, args.output)
    convert_vae_decoder(pipe, args.output)

    print("\n✅ Conversion complete!")
    print(f"Output: {os.path.abspath(args.output)}")
    print("\nNext steps:")
    print("  1. Quantize with: python quantize_model.py --input onnx_models/ --output quantized/")
    print("  2. Push to phone: Android/data/com.localaipainter/files/models/")

if __name__ == "__main__":
    main()
