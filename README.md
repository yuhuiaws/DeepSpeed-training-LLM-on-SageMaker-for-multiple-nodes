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
3. By default, DeepSpeed expects that a multi-node environment uses a shared storage. If this is not the case and each node can only see the local filesystem，you need to set the parameter "save_on_each_node" of Seq2SeqTrainingArguments API or TrainingArguments API to true.
4. When you use deepspeed multiple nodes training and set the parameter "load_best_model_at_end" (from Seq2SeqTrainingArguments or TrainingArguments API) to true, maybe error will happens when finishing training procedure. The error looks like the following: 

Could not locate the best model at /tmp/checkpoint-60/pytorch_model.bin, if you are running a distributed training on multiple nodes, you should activate `--save_on_each_node`. 

In fact, I have configured the parameter "save_on_each_node" to true (my environment: transformer 4.26.0，pytorch 1.10，python 3.8). I will only save best model, configure "save_on_each_node" to false and fix the issue.

5. If you just want to save the best model weights, you can set the parameter "output_dir" (from Seq2SeqTrainingArguments or TrainingArguments API) to temporary path such as '/tmp' on p4d.24xlarge ("/tmp" has the enough disk space to save); And if you want to save all of the checkpoint during the training, you can set the output_dir to the checkponit local path (it will impact the train speed for multi-nodes training. Because SageMaker will upload the checkpoint to S3 nearly real-time, it will occupy the networking bandwidth and impact the communication efficiency between nodes in the cluster).
6. When using parameter "compute_metrics" from Trainer or Seq2SeqTrainer API, the evaluation procedure is very slow. So if you just want to run successfully the whole training process, you can comment out the  "compute_metrics".
7. 















