from termcolor import colored

def input_with_default(prompt:str,default:str):
    prompt=colored(prompt,"yellow",attrs=["bold"])
    default_colord=colored(f"default: [{default}]","green",attrs=["bold"])
    userinput=input(f"{prompt}---{default_colord}\n")
    return userinput or default

x=input_with_default("xxx","123")
print(x)