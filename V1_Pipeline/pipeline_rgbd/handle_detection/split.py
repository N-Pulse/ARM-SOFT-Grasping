import splitfolders   

splitfolders.ratio(
    "cup_handle/images/raw",            
    output="cup_handle/images",         
    seed=1337,
    ratio=(.7, .2, .1)                  
)
