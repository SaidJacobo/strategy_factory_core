
class OperationResult:
    
    def __init__(self, ok:bool, message:str, item:object):
        self.ok = ok
        self.message = message
        self.item = item
        
    
    
    def __repr__(self):
        return f"<Result(ok={self.ok}, message='{self.message}', item={self.item})>"
