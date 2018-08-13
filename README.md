# ATT (Atlas Toolbox)
Atlas Toolbox
ATT is developed to construct brain activity atlas, collect and quantify multimodal characteristics of brain areas.

## Installation and setup
You can download this toolbox from the github page.
Then you need to configure environment variable in your system.
If you're using a linux system, please install it under below step:

1 Click Clone or download, select Download ZIP from the github page [https://github.com/helloTC/ATT](https://github.com/helloTC/ATT).
2 Unzip your downloaded toolbox to one folder (with the directory as <your_directory>).
3 Configure the environment variable using .bashrc, do as:

'''bash
$ gedit ~/.bashrc
'''

In .bashrc, edit it and add:

'''
export PYTHONPATH=$PYTHONPATH:<your_directory>
'''

Exit .bashrc, in your teminal, execute:
'''bash
$ source ~/.bashrc
'''

Now you have installed this toolbox in your python environment.

### Usage
In your python environment, if you'd like to call functions in this toolbox, you can import it as other packages:

'''python
>>> from ATT.algorithms import tools
>>> tools.pearsonr?
>>> tools.pearsonr(A, B)
'''

### How to get involved
I'm thrilled to welcome new contributors! If you have good ideas on the framework or some specific functions, please contact with me:
[taicheng_huang@mail.bnu.edu.cn](taicheng_huang@mail.bnu.edu.cn)
