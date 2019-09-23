import yaml
import redis


class Config:
    def __init__(self, file_path=None):
        if file_path:
            with open(file_path) as f:
                config = yaml.load(f)
        else:
            config = {'data': {'host': None, 'port': None}}
        if not config['data']['host']:
            config['data']['host'] = "localhost"
        if not config['data']['port']:
            config['data']['port'] = "6379"
        self.db = redis.StrictRedis(host=config['data']['host'],
                                    port=config['data']['port'], db=0)

