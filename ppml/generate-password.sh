mkdir password && cd password
export YOUR_PASSWORD=$1 #path_of_your_pwd
openssl genrsa -out key.txt 2048
echo "YOUR_PASSWORD" | openssl rsautl -inkey key.txt -encrypt >output.bin
