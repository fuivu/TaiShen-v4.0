#!/usr/bin/env python3
"""
Quantize ONNX models to INT8 using ONNX Runtime quantization tools.

Usage:
    python quantize_model.py --input ./onnx_models/ --output ./quantized/
"""

import argparse
import os
import glob
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_file(input_path, output_path, quant_type=QuantType.QInt8):
    """Quantize a single ONNX file"""
    print(f"  Quantizing: {os.path.basename(input_path)}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=quant_type
    )
    orig_size = os.path.getsize(input_path) / (1024*1024)
    quant_size = os.path.getsize(output_path) / (1024*1024)
    ratio = orig_size / quant_size if quant_size > 0 else 0
    print(f"    {orig_size:.1f}MB → {quant_size:.1f}MB (compression: {ratio:.1f}x)")

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument("--input", default="./onnx_models", help="Input directory")
    parser.add_argument("--output", default="./quantized", help="Output directory")
    parser.add_argument("--type", choices=["int8", "int4"], default="int8")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(os.path.join(args.input, "*.onnx"))
    if not files:
        print(f"No ONNX files found in {args.input}")
        return

    print(f"Found {len(files)} ONNX files")
    quant_type = QuantType.QInt8 if args.type == "int8" else QuantType.QInt4

    for f in files:
        basename = os.path.basename(f)
        output_path = os.path.join(args.output, basename)
        quantize_file(f, output_path, quant_type)

    print("\n✅ Quantization complete!")
    print(f"Output: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
