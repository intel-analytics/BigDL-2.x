from zoo.serving.client import InputQueue, OutputQueue


class TestClient:

    def test_input_queue(self):
        input_api = InputQueue(host="1.1.1.1", port="1111", name="my-test")
        assert input_api.name == "my-test"
        assert input_api.host == "1.1.1.1"
        assert input_api.port == "1111"

    def test_output_queue(self):
        output_api = OutputQueue(host="1.1.1.1", port="1111", name="my-test")
        assert output_api.name == "my-test"
        assert output_api.host == "1.1.1.1"
        assert output_api.port == "1111"
