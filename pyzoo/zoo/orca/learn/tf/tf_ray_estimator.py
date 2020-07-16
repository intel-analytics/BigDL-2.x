from zoo.orca.learn.tf.tf_trainer import TFTrainer
from zoo.ray import RayContext

class TFRayEstimator():
    
    def __init__(self,
                 model_creator,
                 compile_args_creator,
                 data_creator,
                 config=None,
                 verbose=False
                ):
        
        def trainer_model_creator(config):
            model = model_creator(config)
            compile_args = compile_args_creator(config),
            # to support horovod, we need to wrap hvd.DistributedOptimizer on
            # compile_args["optimizer"]
            model.compile(**compile_args)
            return model
        self.ray_ctx = RayContext.get()
        self.tf_trainer = TFTrainer(trainer_model_creator,
                                    data_creator,
                                    config=config,
                                    num_replicas=self.num_ray_nodes,
                                    num_cpus_per_worker=self.ray_ctx.ray_node_cpu_cores,
                                    use_gpu=False,
                                    verbose=verbose)
    def train(self):
        return self.tf_trainer.train()

    def validate(self):
        return self.tf_trainer.validate()

    def get_model(self):
        return self.tf_trainer.get_model()

    def save(self, checkpoint):
        return self.tf_trainer.save()

    def restore(self, checkpoint):
        self.tf_trainer.restore()

    def shutdown(self):
        self.tf_trainer.shutdown()