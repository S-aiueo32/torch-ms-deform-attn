{
  description = "torch-ms-deform-attn Kernel Hub adapter";
  inputs.kernel-builder.url = "github:huggingface/kernels/v0.16.1";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
