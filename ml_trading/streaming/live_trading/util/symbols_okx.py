import requests

_flag = "0"  # live trading: 0, demo trading: 1

def _get_swap_instruments():
    r = requests.get("https://www.okx.com/api/v5/public/instruments?instType=SWAP")
    r_js = r.json()
    
    return r_js


def get_swap_symbobls_usd():
    result_get_instruments = _get_swap_instruments()

    inst_ids = [instData['instId'] for instData in result_get_instruments['data']]
    inst_ids_usd = [inst_id for inst_id in inst_ids if 'USD-' in inst_id]

    return inst_ids_usd



def get_swap_symbobls_usdc():
    result_get_instruments = _get_swap_instruments()

    inst_ids = [instData['instId'] for instData in result_get_instruments['data']]
    inst_ids_usdc = [inst_id for inst_id in inst_ids if 'USDC-' in inst_id]

    return inst_ids_usdc


def get_swap_symbobls_usdt():
    result_get_instruments = _get_swap_instruments()

    inst_ids = [instData['instId'] for instData in result_get_instruments['data']]
    inst_ids_usdt = [inst_id for inst_id in inst_ids if 'USDT-' in inst_id]

    return inst_ids_usdt


def get_swap_symbobls_usd_all():
    return get_swap_symbobls_usd() + get_swap_symbobls_usdc() + get_swap_symbobls_usdt()

