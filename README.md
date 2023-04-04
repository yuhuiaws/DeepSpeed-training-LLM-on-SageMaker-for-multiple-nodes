# DeepSpeed-training-LLM-on-AWS-SageMaker-for-multiple-nodes-and-deploy-on-AWS-SageMaker

This repo will show the whole codes:
1. Fine tuning LLM by DeepSpeed on SageMaker for multiple nodes.
2. Deploy the trained model from above step #1 on SageMaker.

The repo is tested successfully on Data Science image and Python 3 kernel of Sagemaker studio with ml.m5.large kernel gateway instance in us-east-1 region (If you encounter with kerenl restaring issue when preparing dataset in DeepSpeed-Flan-T5-on-Sagemaker-multiple-nodes.ipynb, I suggest that you shut down the kernel gateway instance and re-execute the DeepSpeed-Flan-T5-on-Sagemaker-multiple-nodes.ipynb).

Fine tuning LLM such as Flan-T5-XXL

Now, we utilize the torch.distributed.launch + Deepspeed + Huggingface trainer API to fine tunig Flan-T5-XXL on AWS SageMaker for multiple nodes. You can follow up the folder structure, and prepare your training script and configure related parameters in the torch_launch.sh script. If you also use the HF high level trainer API to train CausalLM (such as GPT-J) or Seq2seqLM (such as T5), there is very little code that needs to be modified.

I explain more about these files: start.py as user entry point will set some environment variables such as master's IP address and invoke the torch_launch.sh script. Most of parameters (including training parameters and torch distributed launcher parameters) should be configured in torch_launch.sh. Finally torch_launch.sh will invoke your training python script. Also, you can use the requirements.txt to install related python libraries.

Some useful tips:
1. There is the open source "s5cmd" file in this repo, we can use the "s5cmd" command to speedup the uploading model assets to S3 (do not tar and compress these model assets, just directly upload to S3) after saving model in the container's local path.
2. When using deepspeed zero stage 2 training LLM on muliple nodes in SageMaker, maybe it will hung untile the NCCL communication is timeout. When it happens, you can check the GPU memory utility of training instances from Amazon cloudwatch. In my experiment, the GPU memory utility is almost full (but OOM didn't occur), it may be a signal that you should switch to zero stage 3 (the issue disappears when I switch to zero 3).
3. By default, DeepSpeed expects that a multi-nodes environment uses a shared storage. If this is not the case and each node can only see the local filesystem，you need to set the parameter "save_on_each_node" of Seq2SeqTrainingArguments API or TrainingArguments API to true (in this repo, I didn't use share data store such as EFS to save model, so I set the "save_on_each_node" to true).
4. When you use deepspeed multiple nodes training and set the parameter "load_best_model_at_end" (from Seq2SeqTrainingArguments or TrainingArguments API) to true, maybe error will happens when finishing training procedure. The error looks like the following: 

        Could not locate the best model at /tmp/checkpoint-60/pytorch_model.bin, if you are running 
        distributed training on multiple nodes, you should activate `--save_on_each_node`. 

  In fact, I have configured the parameter "save_on_each_node" to true (my environment: transformer 4.26.0，pytorch 1.10，python 3.8). I will only save best model, configure "load_best_model_at_end" to false and fix the issue.

5. If you just want to save the best model weights, you can set the parameter "output_dir" (from Seq2SeqTrainingArguments or TrainingArguments API) to temporary path such as '/tmp' on p4d.24xlarge ("/tmp" has the enough disk space to save); And if you want to save all of the checkpoint during the training, you can set the output_dir to the checkponit local path (it will impact the train speed for multi-nodes training. Because SageMaker will upload the checkpoint to S3 nearly real-time, it will occupy the networking bandwidth and impact the communication efficiency between nodes in the cluster).
6. When using parameter "compute_metrics" from Trainer or Seq2SeqTrainer API, the evaluation procedure is very slow. So if you just want to run successfully the whole training process, you can comment out the  "compute_metrics".
7. When your training script will download something from website (such as nltk.downlaod("punkt")), you should ensure only one process in the current node (local rank 0) downloaindg files, otherwise it may fail the training job. 

        Traceback (most recent call last):
          File "/opt/ml/code/T5_configz_and_code/scripts/run_seq2seq_deepspeed.py", line 26, in <module>
            nltk.download("punkt", quiet=True)
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 777, in download
        for msg in self.incr_download(info_or_id, download_dir, force):
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 642, in incr_download
        yield from self._download_package(info, download_dir, force)
          File "/opt/conda/lib/python3.8/site-packages/nltk/downloader.py", line 699, in _download_package
        os.makedirs(download_dir)
          File "/opt/conda/lib/python3.8/os.py", line 223, in makedirs
        mkdir(name, mode)
        FileExistsError: [Errno 17] File exists: '/root/nltk_data'
        [nltk_data] [Errno 2] No such file or directory:
        [nltk_data]     '/root/nltk_data/tokenizers/punkt.zip'
        [nltk_data] Error with downloaded zip file
        [nltk_data] [Errno 2] No such file or directory:
        [nltk_data]     '/root/nltk_data/tokenizers/punkt.zip'
        Downloading builder script:   0%|          | 0.00/6.27k [00:00<?, ?B/s]
        Downloading builder script: 100%|██████████| 6.27k/6.27k [00:00<00:00, 7.75MB/s]
        [nltk_data] Error with downloaded zip file
        [nltk_data] Error with downloaded zip file
        [nltk_data] Error with downloaded zip file

If you use the torch.distributed.launch, you can utilize the barrier function to achivement this purpose. More details, you can find the related code from run_seq2seq_deepspeed.py.

8. When you use torch.distributed.launch, please don't use global variables in your training script. Otherwise, the CUDA errors may occurs when exiting your training script and fails the training job. So in run_seq2seq_deepspeed.py, I change the "metric" variable from global variable to local variable.
9. Plesae do not save the model into "/opt/ml/model", because Sagemaker will tar and compress all of files under "/opt/ml/model", and it will consume much time for LLM). I suggest that '/tmp/output/asset/' can be used to perform the model saving.
10. We just use the rank 0 process to upload the trained model assets to S3 by s5cmd command. It means just one of ranks will perform the thing even if multiple nodes training is used.
11. We should sync with every rank and ensure rank 0 uploading the model assets successfully (putting the torch.distributed.barrier() at the end of your taining script). Ater that, maybe there is some CUDA error when exiting the process:

          terminate called after throwing an instance of 'c10::CUDAError'
            what():  CUDA error: driver shutting down
          CUDA kernel errors might be asynchronously reported at some other API call,
          so the stacktrace below might be incorrect.
          For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
          Exception raised from query at ../aten/src/ATen/cuda/CUDAEvent.h:95 (most recent call first):

Please just ignore the error because the trained model assets have been uploaded to the S3.
 

Deploy LLM on SageMaker

Now, we suggest that trained LLM is deployed by LMI (large model inference) container on SageMaker. LMI support 3 types accelerator: huggingface accelerate, deepspeed inference, faster transformer.

















