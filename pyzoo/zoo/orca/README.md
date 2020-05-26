## Project Orca: Easily Scaling out Python AI pipelines

Most AI projects start with a Python notebook running a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger data set in a distributed fashion. 

_Project Orca_ allows you to easily scale out your single node Python notebook across large clusters, by providing:
* Data-parallel preprocessing for AI (supporting common Python libraries such as Pandas, Numpy, PIL, Tensorflow Dataset, PyTorch Dataloader, etc.) 
* Sklearn-style APIs for distributed training and inference (supporting TensroFlow, PyTorch, Keras, MXNet, Horovod, etc.)
