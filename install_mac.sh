#!/bin/bash
file="./config/python_path.txt"
python_exec=""

while IFS= read -r line; do
python_exec+=" $line"
done < "$file"

echo $python_exec

requirements_txt="./config/requirements_mac.txt"

echo "Installing ComfyUI's Mixlab Nodes.."

if [ -x "$(command -v python)" ]; then
echo "Installing with system Python"
while IFS= read -r line; do
pip install "$line" -i https://pypi.tuna.tsinghua.edu.cn/simple
done < "$requirements_txt"
else
echo "Installing with ComfyUI Portable"
while IFS= read -r line; do
python -m pip install "$line" -i https://pypi.tuna.tsinghua.edu.cn/simple
done < "$requirements_txt"
fi
read -p "Press Enter to exit"