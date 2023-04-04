# DeepSpeed-training-LLM-on-AWS-SageMaker-for-multiple-nodes-and-deploy-on-AWS-SageMaker

This repo will show the whole codes:
1. Fine tuning LLM by DeepSpeed on SageMaker for multiple nodes.
2. Deploy the trained model for above step #1 on SageMaker.

Fine tuning LLM such as Flan-T5-XXL

Now, we utilize the torch.distributed.launch + Deepspeed + Huggingface trainer API to fine tunig Flan-T5-XXL on AWS SageMaker for multiple nodes. You can follow up the folder structure, and prepare your training script and configure related parameters in the torch_launch.sh script. If you also use the HF high level trainer API to train CausalLM (such as GPT-J) or Seq2seqLM (such as T5), there is very little code that needs to be modified.

I explain more about these files: start.py will set some environment variables such as master's IP address and invoke the torch_launch.sh. Most of parameters (including training parameters and torch distributed launcher parameters) should be configured in torch_launch.sh. Finally torch_launch.sh will invoke your training python script. Also, you can use the requirements.txt to install related python libraries.

Some tips:
1. There is the "s5cmd" file in this repo, we can use the command to speedup the uploading model assets to S3 after saving model in the container's local path.
2. When using deepspeed zero stage 2 training LLM on muliple nodes in SageMaker, maybe it will hung untile the NCCL communication is timeout. When it happens, you can check the GPU memory utility of training instances from Amazon cloudwatch. In my experiment, the GPU memory utility is almost full (but OOM didn't occur), it may be a signal that you should switch to zero stage 3 (the issue disappears when I switch to zero 3).
3. 










