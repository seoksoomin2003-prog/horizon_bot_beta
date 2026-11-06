from typing import Dict, Any, Optional
from adapters.base import AbstractExchangeAdapter

class BinanceAdapter(AbstractExchangeAdapter):
    def format_order_params(self, symbol: str, side: str, amount: float, price: float, client_order_id: str, reduce_only: bool = False, order_type: str = "limit", params: Dict[str, Any] = None) -> Dict[str, Any]:
        p = params or {}
        if client_order_id:
            p['newClientOrderId'] = client_order_id
        if reduce_only:
            p['reduceOnly'] = True
        return p

    def parse_plan_order(self, response: Dict[str, Any]) -> str:
        return response.get('id') or response.get('orderId') or str(response)

    def get_position_mode(self, client: Any) -> str:
        try:
            res = client.fapiPrivateGetPositionSideDual()
            if res and isinstance(res, dict) and str(res.get('dualSidePosition')).lower() == 'true':
                return 'hedge'
        except Exception:
            pass
        return 'oneway'

    def format_trade_side(self, params: Dict[str, Any], side: str, position_mode: str, is_close: bool = False) -> Dict[str, Any]:
        if position_mode == 'hedge':
            normalized = side.upper()
            if normalized in ['LONG', 'BUY']:
                params['positionSide'] = 'LONG'
            elif normalized in ['SHORT', 'SELL']:
                params['positionSide'] = 'SHORT'
            else:
                params['positionSide'] = normalized
        elif is_close:
            params['reduceOnly'] = True
        return params

    def patch_exchange(self, client: Any) -> None:
        import ccxt
        import os
        import json
        import time
        import threading
        from logger import trades_logger, errors_logger
        from utils.persistence import get_absolute_path, atomic_write_json, ProcessLock
        
        ccxt_version = getattr(ccxt, '__version__', 'unknown')
        trades_logger.info(f"Applying Binance Unified Order patch. CCXT Version: {ccxt_version}")
        
        if not hasattr(BinanceAdapter, '_registry_thread_lock'):
            BinanceAdapter._registry_thread_lock = threading.RLock()

        # Persistent registry for deterministic algo order discovery
        registry_path = get_absolute_path("algo_registry.json")

        def load_registry():
            if os.path.exists(registry_path):
                try:
                    with open(registry_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    errors_logger.error(f"Failed to load algo registry: {e}")
            return {}

        if not hasattr(client, '_algo_registry'):
            raw_reg = load_registry()
            meta = raw_reg.pop('_metadata', {'version': 0})
            client._algo_registry = raw_reg
            client._registry_version = meta.get('version', 0)

        def save_registry():
            with BinanceAdapter._registry_thread_lock:
                lock_path = get_absolute_path("algo_registry.lock")
                proc_lock = ProcessLock(lock_path)
                
                acquired = False
                for _ in range(50):  # 5 seconds max wait
                    if proc_lock.acquire():
                        acquired = True
                        break
                    time.sleep(0.1)
                    
                if not acquired:
                    errors_logger.error("Failed to acquire process lock for algo registry save")
                    return
                    
                try:
                    disk_state = load_registry()
                    disk_meta = disk_state.pop('_metadata', {'version': 0})
                    disk_version = disk_meta.get('version', 0)
                    
                    new_version = max(disk_version, client._registry_version) + 1
                    
                    merged = {}
                    all_keys = set(disk_state.keys()) | set(client._algo_registry.keys())
                    
                    for k in all_keys:
                        d_obj = disk_state.get(k)
                        l_obj = client._algo_registry.get(k)
                        
                        if d_obj and not l_obj:
                            merged[k] = d_obj
                        elif l_obj and not d_obj:
                            merged[k] = l_obj
                        else:
                            d_gen = d_obj.get('generation', 0)
                            l_gen = l_obj.get('generation', 0)
                            d_del = d_obj.get('deleted', False)
                            l_del = l_obj.get('deleted', False)
                            
                            if not d_del and not l_del:
                                merged[k] = l_obj if l_gen >= d_gen else d_obj
                            elif d_del and l_del:
                                merged[k] = l_obj if l_obj.get('deleted_at', 0) >= d_obj.get('deleted_at', 0) else d_obj
                            elif l_del and not d_del:
                                merged[k] = d_obj if d_gen > l_gen else l_obj
                            elif not l_del and d_del:
                                merged[k] = l_obj if l_gen > d_gen else d_obj
                                
                    # GC Tombstones older than 30 days
                    RETENTION_MS = 30 * 24 * 60 * 60 * 1000
                    current_time = int(time.time() * 1000)
                    for k in list(merged.keys()):
                        obj = merged[k]
                        if obj.get('deleted'):
                            del_at = obj.get('deleted_at', 0)
                            if del_at > 0 and (current_time - del_at) > RETENTION_MS:
                                del merged[k]
                                
                    client._algo_registry = merged
                    client._registry_version = new_version
                    
                    write_state = merged.copy()
                    write_state['_metadata'] = {'version': new_version}
                    
                    atomic_write_json(registry_path, write_state)
                except Exception as e:
                    errors_logger.error(f"Failed to save algo registry: {e}")
                finally:
                    proc_lock.release()

        def register_algo_order(order_data: dict, force_save: bool = True):
            oid = str(order_data.get('id', ''))
            if not oid: return
            
            info = order_data.get('info', {})
            raw_algo_id = str(info.get('algoId') or order_data.get('algoId') or oid)
            
            prev_gen = 0
            if oid in client._algo_registry:
                prev_gen = client._algo_registry[oid].get('generation', 0)
                
            client._algo_registry[oid] = {
                "order_id": oid,
                "algo_id": raw_algo_id,
                "client_algo_id": order_data.get('clientOrderId', ''),
                "symbol": order_data.get('symbol', ''),
                "type": order_data.get('type', ''),
                "side": order_data.get('side', ''),
                "namespace": "ALGO",
                "created_at": order_data.get('timestamp', int(time.time() * 1000)),
                "generation": prev_gen + 1
            }
            if force_save:
                save_registry()

        def resolve_algo_identity(target_id: str) -> str:
            target_str = str(target_id)
            if target_str in client._algo_registry and not client._algo_registry[target_str].get('deleted'):
                return target_str
            for oid, data in client._algo_registry.items():
                if data.get('deleted'): continue
                if data.get('algo_id') == target_str:
                    return oid
                if data.get('client_algo_id') == target_str:
                    return oid
            return target_str

        def remove_algo_id(oid: str):
            resolved = resolve_algo_identity(oid)
            if resolved in client._algo_registry:
                client._algo_registry[resolved]['deleted'] = True
                client._algo_registry[resolved]['deleted_at'] = int(time.time() * 1000)
                save_registry()

        original_create_order = client.create_order
        def create_order_patched(symbol, type, side, amount, price=None, params=None):
            params = params or {}
            order = original_create_order(symbol, type, side, amount, price, params)
            if str(type).upper() in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT', 'TRAILING_STOP_MARKET']:
                if order:
                    register_algo_order(order, force_save=True)
            return order
        client.create_order = create_order_patched

        original_fetch_open = client.fetch_open_orders
        def fetch_open_orders_patched(symbol=None, since=None, limit=None, params=None):
            params = params or {}
            try:
                std_orders = original_fetch_open(symbol, since, limit, params)
            except Exception as e:
                errors_logger.warning(f"Failed to query standard orders: {e}")
                std_orders = []
                
            algo_params = params.copy()
            algo_params['conditional'] = True
            try:
                algo_orders = original_fetch_open(symbol, since, limit, algo_params)
                changed = False
                for o in algo_orders:
                    oid = str(o.get('id', ''))
                    if oid and oid not in client._algo_registry:
                        register_algo_order(o, force_save=False)
                        changed = True
                
                symbol_registry_count = len([v for k, v in client._algo_registry.items() if v.get('symbol') == symbol and not v.get('deleted')])
                needs_recovery = not client._algo_registry or changed or len(algo_orders) > symbol_registry_count
                
                if needs_recovery and symbol and hasattr(client, 'fetch_orders'):
                    try:
                        hist_params = algo_params.copy()
                        MAX_HISTORICAL_PAGES = 10
                        fetched_pages = 0
                        current_since = since
                        
                        while fetched_pages < MAX_HISTORICAL_PAGES:
                            hist_orders = client.fetch_orders(symbol, current_since, limit, hist_params)
                            if not hist_orders:
                                break
                                
                            for o in hist_orders:
                                oid = str(o.get('id', ''))
                                if oid and oid not in client._algo_registry:
                                    register_algo_order(o, force_save=False)
                                    changed = True
                                    
                            last_timestamp = hist_orders[-1].get('timestamp')
                            if not last_timestamp or last_timestamp == current_since:
                                break
                            current_since = last_timestamp + 1
                            fetched_pages += 1
                    except Exception as e:
                        errors_logger.warning(f"Failed to query historical algo orders: {e}")

                if changed:
                    save_registry()
            except ccxt.BadRequest as e:
                errors_logger.warning(f"Algo order endpoint not supported or bad request: {e}")
                algo_orders = []
            except Exception as e:
                errors_logger.error(f"Failed to query algo orders: {e}")
                algo_orders = []
            
            merged = std_orders + algo_orders
            seen = set()
            unique_orders = []
            for o in merged:
                comp_key = f"{o.get('id')}_{o.get('clientOrderId', '')}_{o.get('type', '')}_{o.get('side', '')}"
                if comp_key not in seen:
                    seen.add(comp_key)
                    unique_orders.append(o)
            return unique_orders
        client.fetch_open_orders = fetch_open_orders_patched

        if hasattr(client, 'fetch_order'):
            original_fetch_order = client.fetch_order
            def fetch_order_patched(id, symbol=None, params=None):
                params = params or {}
                resolved_id = resolve_algo_identity(id)
                if resolved_id in client._algo_registry:
                    algo_params = params.copy()
                    algo_params['conditional'] = True
                    return original_fetch_order(resolved_id, symbol, algo_params)
                    
                try:
                    return original_fetch_order(resolved_id, symbol, params)
                except ccxt.OrderNotFound as e:
                    algo_params = params.copy()
                    algo_params['conditional'] = True
                    try:
                        order = original_fetch_order(resolved_id, symbol, algo_params)
                        if order:
                            register_algo_order(order, force_save=True)
                        return order
                    except Exception:
                        raise e
            client.fetch_order = fetch_order_patched

        if hasattr(client, 'cancel_order'):
            original_cancel = client.cancel_order
            def cancel_order_patched(id, symbol=None, params=None):
                params = params or {}
                resolved_id = resolve_algo_identity(id)
                if resolved_id in client._algo_registry:
                    params['conditional'] = True
                    
                try:
                    res = original_cancel(resolved_id, symbol, params)
                    remove_algo_id(resolved_id)
                    return res
                except ccxt.OrderNotFound as e:
                    if not params.get('conditional'):
                        algo_params = params.copy()
                        algo_params['conditional'] = True
                        try:
                            res = original_cancel(resolved_id, symbol, algo_params)
                            remove_algo_id(resolved_id)
                            return res
                        except Exception:
                            raise e
                    raise e
            client.cancel_order = cancel_order_patched

        if hasattr(client, 'cancel_all_orders'):
            original_cancel_all = client.cancel_all_orders
            def cancel_all_orders_patched(symbol=None, params=None):
                params = params or {}
                results = []
                try:
                    res1 = original_cancel_all(symbol, params)
                    results.append(res1)
                except Exception:
                    pass
                algo_params = params.copy()
                algo_params['conditional'] = True
                try:
                    res2 = original_cancel_all(symbol, algo_params)
                    results.append(res2)
                    if symbol is None:
                        for k in client._algo_registry.keys():
                            client._algo_registry[k]['deleted'] = True
                            client._algo_registry[k]['deleted_at'] = int(time.time() * 1000)
                    else:
                        for k, v in client._algo_registry.items():
                            if v.get('symbol') == symbol:
                                client._algo_registry[k]['deleted'] = True
                                client._algo_registry[k]['deleted_at'] = int(time.time() * 1000)
                    save_registry()
                except Exception:
                    pass
                return results
            client.cancel_all_orders = cancel_all_orders_patched
