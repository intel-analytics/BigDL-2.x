## PPML (Privacy Preserving Machine Learning) 
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs. <br>

### Create SGX driver
```bash
./install-graphene-driver.sh
```

### Generate keys
```bash
./generate-keys.sh
```

### Generate password
```bash
./generate-password.sh used_password_in_generate-keys.sh
```
For example: <br>
`./generate-password.sh abcd1234`
