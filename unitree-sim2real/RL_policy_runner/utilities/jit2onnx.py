import os
import sys
import torch
import argparse

def convert_jit_to_onnx(model_path, input_dim=42, output_path=None):
    """
    Convert a PyTorch JIT model to ONNX format.
    
    Args:
        model_path: Path to the .pt JIT model file
        input_dim: Input dimension for the dummy input (default: 42)
        output_path: Optional output path for the ONNX file. If None, saves alongside the .pt file
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False
    
    if not model_path.endswith('.pt'):
        print(f"Error: Expected a .pt file, got '{model_path}'")
        return False
    
    # Determine output path
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    print(f"Loading JIT model from: {model_path}")
    try:
        scripted_model = torch.jit.load(model_path)
        scripted_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_dim)
        
        print(f"Converting to ONNX with input dimension: {input_dim}")
        torch.onnx.export(
            scripted_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        
        print(f"Successfully converted to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch JIT model to ONNX format')
    parser.add_argument('--model-path', '--model_path', type=str, required=True, help='Path to the .pt JIT model file')
    parser.add_argument('--input-dim', type=int, default=42, help='Input dimension for the model (default: 42)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for the ONNX file (default: same as input with .onnx extension)')
    
    args = parser.parse_args()
    
    success = convert_jit_to_onnx(args.model_path, args.input_dim, args.output)
    sys.exit(0 if success else 1)
