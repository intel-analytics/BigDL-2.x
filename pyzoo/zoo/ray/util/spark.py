import os


class SparkRunner():
    def __init__(self):
        import pyspark
        print("Current pyspark location is : {}".format(pyspark.__file__))

    # This is adopted from conda pack.
    def _pack_conda_main(self, args):
        import sys
        import traceback
        from conda_pack.cli import fail, PARSER, context
        import conda_pack
        from conda_pack import pack, CondaPackException
        args = PARSER.parse_args(args=args)
        # Manually handle version printing to output to stdout in python < 3.4
        if args.version:
            print('conda-pack %s' % conda_pack.__version__)
            sys.exit(0)

        try:
            with context.set_cli():
                pack(name=args.name,
                     prefix=args.prefix,
                     output=args.output,
                     format=args.format,
                     force=args.force,
                     compress_level=args.compress_level,
                     n_threads=args.n_threads,
                     zip_symlinks=args.zip_symlinks,
                     zip_64=not args.no_zip_64,
                     arcroot=args.arcroot,
                     dest_prefix=args.dest_prefix,
                     verbose=not args.quiet,
                     filters=args.filters)
        except CondaPackException as e:
            fail("CondaPackError: %s" % e)
        except KeyboardInterrupt:
            fail("Interrupted")
        except Exception:
            fail(traceback.format_exc())

    def pack_penv(self, conda_name):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = "{}/python_env.tar.gz".format(tmp_dir)
        print("Start to pack current python env")
        self._pack_conda_main(["--output", tmp_path, "--n-threads", "8", "--name", conda_name])
        print("Packing has been completed: {}".format(tmp_path))
        return tmp_path

    def _common_opt(self, master):
        return '--master {} '.format(master)

    def _create_sc(self, submit_args, conf):
        from pyspark.sql import SparkSession
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args
        spark_conf = SparkSession.builder
        for key, value in conf.items():
            spark_conf.config(key=key, value=value)
        # .config(key="spark.executor.pyspark.memory", value=spark_executor_pyspark_memory)
        spark = spark_conf.getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("INFO")

        return sc

    def init_spark_on_local(self,
                   python_loc,
                   driver_memory,
                   master=None):
        # os.environ['PYSPARK_PYTHON'] = session_execute("which python").out
        # TODO: remove python_loc, "which python" would generate /usr/bin/python
        os.environ['PYSPARK_PYTHON'] = python_loc
        sc = self._create_sc(self._common_opt(master) + 'pyspark-shell',
                        {"spark.driver.memory": driver_memory})
        return sc


    def _init_yarn(self,
                   python_zip_file,
                   driver_memory,
                   driver_cores,
                   master=None,
                   # settings for cluster mode
                   executor_cores=None,
                   executor_memory=None,
                   extra_executor_memory_for_ray=None,
                   #  settings for yarn only
                   num_executor=None,
                   spark_yarn_jars=None,
                   penv_archive=None,
                   hadoop_conf=None,
                   hadoop_user_name=None):
        os.environ["HADOOP_CONF_DIR"] = hadoop_conf
        os.environ['HADOOP_USER_NAME'] = hadoop_user_name
        os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"

        if not python_zip_file:
            python_zip_file = ""

        def _yarn_opt():
            return " --archives {}#python_env --num-executors {} " \
                   " --executor-cores {} --executor-memory {} --py-files {}  ".format(
                penv_archive, num_executor, executor_cores, executor_memory, python_zip_file)

        def _submit_opt(master):
            conf = {
                "spark.driver.memory": driver_memory,
                "spark.driver.cores": driver_cores,
                "spark.scheduler.minRegisterreResourcesRatio": "1.0",
                "spark.task.cpus": executor_cores}
            if extra_executor_memory_for_ray:
                conf["spark.executor.memoryOverhead"] = extra_executor_memory_for_ray
            if spark_yarn_jars:
                conf.insert("spark.yarn.archive", spark_yarn_jars)
            return self._common_opt(master) + _yarn_opt() + 'pyspark-shell', conf

        submit_args, conf = _submit_opt(master)
        return self._create_sc(submit_args, conf)

    def init_spark_on_yarn(self,
                           hadoop_conf,
                           extra_pmodule_zip,
                           num_executor,
                           conda_name,
                           executor_cores,
                           executor_memory="10g",
                           driver_memory="1g",
                           driver_cores=10,
                           extra_executor_memory_for_ray=None,
                           penv_archive=None,
                           master="yarn",
                           hadoop_user_name="root",
        spark_yarn_jars=None):
        pack_env = False
        assert penv_archive or conda_name,\
            "You should either specify penv_archive or conda_name explicitly"
        try:
            if not penv_archive:
                penv_archive = self.pack_penv(conda_name)
                pack_env = True
            sc = self._init_yarn(hadoop_conf=hadoop_conf,
                              spark_yarn_jars=spark_yarn_jars,
                              penv_archive=penv_archive,
                              python_zip_file=extra_pmodule_zip,
                              num_executor=num_executor,
                              executor_cores=executor_cores,
                              executor_memory=executor_memory,
                              driver_memory=driver_memory,
                              driver_cores=driver_cores,
     extra_executor_memory_for_ray=extra_executor_memory_for_ray,
                              master=master,
                              hadoop_user_name=hadoop_user_name)
        finally:
            if conda_name and penv_archive and pack_env:
                os.remove(penv_archive)
        return sc

