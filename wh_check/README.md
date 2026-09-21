vLLM 0.27.1 (cu129) wheelhouse + PR#52816 DFlash2 overlay for the ARC-AGI-3
duck harness. Local version tags stripped from wheel filenames (Kaggle mangles
'+'). overlay/ contains the 7 pure-python files from vllm-project/vllm#52816
(DFlash2 draft-model support) applied on top of the 0.27.1 release; the kernel
copies them over vllm-site-packages after install.
