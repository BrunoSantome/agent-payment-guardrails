class PaymentStore:
    def store_payment(self):
        raise NotImplementedError

    def update_status_payment(self):
        raise NotImplementedError

    def store_request(self, request):
        raise NotImplementedError


from collections import defaultdict


# One inMemory store for each beneficiary. Not scalable just dev
class PaymentStoreInMemory(PaymentStore):
    def __init__(
        self,
        beneficiary,
    ):
        self.receipts = {}
        self.requests = []
        self.beneficiary = beneficiary

    def store_payment(self): ...
    def update_status_payment(self): ...
    def store_request(self, request): ...
