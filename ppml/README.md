## PPML (Privacy Preserving Machine Learning) 
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs. <br>

### Create SGX driver
```bash
./install-graphene-driver.sh
```

### Generate keys
The ppml in analytics zoo need secured keys, you need to prepare the secure keys and keystores. <br>
```bash
./generate-keys.sh
```

### Generate password
You also need to store the password you used in previous step in a secured file: <br>
```bash
./generate-password.sh used_password_in_generate-keys.sh
```
For example: <br>
`./generate-password.sh abcd1234`
