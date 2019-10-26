def log (msg, length=False):
    if length == True:
        print(40 * '-' + '\n {}'.format(msg))
    else:
        print(msg)