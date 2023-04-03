# DeepSpeed-training-LLM-on-AWS-SageMaker-for-multiple-nodes-and-deploy-on-AWS-SageMaker

This repo will show the whole codes:
1. Fine tuning LLM by DeepSpeed on SageMaker for multiple nodes.
2. Deploy the trained model for above step #1 on SageMaker.

Fine tuning LLM such as Flan-T5-XXL
Now, we utilize the torch.distributed.launch + Deepspeed + Huggingface trainer API to fine tunig Flan-T5-XXL on AWS SageMaker for multiple nodes.





