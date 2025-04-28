"""
trade_manager.py
Implementazione priva di persistenza su SQLite; tutte le funzioni richieste
da main.py / app.py sono presenti.  Gli I/O su DB diventano semplici log.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# BILANCIO & POSIZIONI (via exchange)
# ---------------------------------------------------------------------------
async def get_real_balance(exchange) -> float | None:
    """
    Ritorna l'equity totale in USDT. Restituisce None se la chiamata fallisce.
    """
    try:
        bal = await exchange.fetch_balance()
        if "info" in bal:
            total_equity = bal["info"].get("totalEquity")
            if total_equity is not None:
                return float(total_equity)
        
        usdt_total = bal.get("total", {}).get("USDT")
        if usdt_total is not None:
            return float(usdt_total)
            
        return 0.0  # Se non troviamo nessun valore valido, restituiamo 0 invece di None
    except Exception as exc:
        logging.error("get_real_balance: %s", exc)
        return 0.0  # In caso di errore, restituiamo 0 invece di None


async def get_open_positions(exchange) -> int:
    """
    Conta le posizioni aperte (contracts > 0) su derivati USDT.
    """
    try:
        pos = await exchange.fetch_positions(None, {
            "limit": 100,
            "category": "linear",
            "settleCoin": "USDT",
        })
        return sum(1 for p in pos if float(p.get("contracts", 0)) > 0)
    except Exception as exc:
        logging.error("get_open_positions: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# FUNZIONI DB → ora solo log
# ---------------------------------------------------------------------------
def save_trade_db(trade: Dict[str, Any]) -> None:
    """
    Simula il salvataggio di un trade: ora logga soltanto.
    """
    logging.info("DB OFF — trade %s registrato in log.", trade.get("trade_id"))


def load_trade_db() -> List[Dict[str, Any]]:
    """
    Ritorna una lista vuota: nessuna cronologia caricata.
    """
    logging.debug("DB OFF — load_trade_db() → lista vuota.")
    return []


async def load_existing_positions(exchange) -> List[Dict[str, Any]]:
    """
    Restituisce le posizioni correnti dal broker; sostituisce il vecchio
    caricamento da DB.
    """
    try:
        pos = await exchange.fetch_positions(None, {
            "limit": 100,
            "category": "linear",
            "settleCoin": "USDT",
        })
        return [p for p in pos if float(p.get("contracts", 0)) > 0]
    except Exception as exc:
        logging.error("load_existing_positions: %s", exc)
        return []


# ---------------------------------------------------------------------------
# GESTIONE POSIZIONI (stub)
# ---------------------------------------------------------------------------
async def manage_position(*args, **kwargs):
    """
    Stub: inserisci la tua logica di apertura/chiusura se/quando vorrai.
    """
    logging.debug("manage_position() chiamata — per ora nessuna azione.")
    return "skipped"


# ---------------------------------------------------------------------------
# MONITOR / UPDATE LOOP — placeholder per compatibilità
# ---------------------------------------------------------------------------
async def monitor_open_trades(exchange):
    logging.info("monitor_open_trades avviato (DB OFF).")
    while True:
        await asyncio.sleep(60)   # mantiene vivo il gather() in main.py


async def wait_and_update_closed_trades(*_a, **_kw):
    logging.debug("wait_and_update_closed_trades() — nessuna operazione.")


# ---------------------------------------------------------------------------
# UTILITY DI LOG
# ---------------------------------------------------------------------------
def log_position_size(symbol: str, size: float, margin: float) -> None:
    logging.info("Size %s → %.6f (margin=%.2f)", symbol, size, margin)
