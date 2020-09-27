
def find_latest_checkpoint(model_dir, model_type="bigdl"):
    import os
    import re
    import datetime
    ckpt_path = None
    latest_version = None
    optim_prefix = None
    optim_regex = None
    if model_type == "bigdl":
        optim_regex = "Sequential[0-9a-z]{8}\.[0-9]+$"
    elif model_type == "pytorch":
        optim_regex = "TorchModel[0-9a-z]{8}\.[0-9]+$"
    elif model_type == "tf":
        optim_regex = "TFParkTraining\.[0-9]+$"
    else:
        ValueError("Only bigdl, pytorch and tf are supported for now.")
    for (root, dirs, files) in os.walk(model_dir, topdown=True):
        temp_versions = []
        timestamps = []
        prefix = None
        for dir in dirs:
            if re.match('(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$', dir) is not None:
                try:
                    # check if dir name is date time
                    datetime.datetime.strptime(dir, '%Y-%m-%d_%H-%M-%S')
                    timestamps.append(dir)
                except:
                    continue
        if timestamps:
            start_dir = os.path.join(root, max(timestamps))
            return find_latest_checkpoint(start_dir)
        for file_name in files:
            if re.match("^optimMethod-" + optim_regex, file_name) is not None:
                file_split = file_name.split(".")
                version = int(file_split[1])
                temp_versions.append(version)
                prefix = file_split[0]
        if temp_versions:
            ckpt_path = root
            latest_version = max(temp_versions)
            optim_prefix = prefix
            break
    return ckpt_path, optim_prefix, latest_version
