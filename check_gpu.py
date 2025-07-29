import torch
import sys

def check_gpu_access():
    print("PyTorch version:", torch.__version__)
    print("Python version:", sys.version)
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List all available GPUs
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
        # Current GPU
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {current_device}")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"Memory allocated: {memory_allocated:.2f} GB")
        print(f"Memory reserved: {memory_reserved:.2f} GB")
        
        # Test tensor creation on GPU
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✓ Successfully created tensor on GPU")
            print(f"Tensor device: {test_tensor.device}")
            del test_tensor  # Clean up
        except Exception as e:
            print(f"✗ Error creating tensor on GPU: {e}")
            
    else:
        print("No CUDA GPUs detected")
        print("Available devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 0 else "None")
    
    # Check MPS (Apple Silicon) support
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\nMPS (Apple Silicon) available: True")
        try:
            test_tensor = torch.randn(100, 100).to('mps')
            print("✓ Successfully created tensor on MPS")
            del test_tensor
        except Exception as e:
            print(f"✗ Error with MPS: {e}")
    else:
        print("\nMPS (Apple Silicon) available: False")

if __name__ == "__main__":
    check_gpu_access()