class Error(Exception): pass
class LeaseClient: default_service_name='lease'
class LeaseKeepAlive:
    def __init__(self,*a,**k): pass
    def is_alive(self): return True
    def shutdown(self): pass
