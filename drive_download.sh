#https://drive.google.com/file/d/1EJ99cADV7PgPYbnsRah0an_IiYrTp61-/view?usp=sharing
fileId=1EJ99cADV7PgPYbnsRah0an_IiYrTp61-
fileName=lmd_matched.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
