class GatewayException(Exception):
    pass


class ProviderNotFound(GatewayException):
    pass


class ProviderTimeout(GatewayException):
    pass


class ProviderUnavailable(GatewayException):
    pass


class InvalidRequest(GatewayException):
    pass
