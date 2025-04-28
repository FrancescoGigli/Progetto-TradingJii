# dependencies.py
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()

import ccxt.async_support as ccxt_async
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time

from config import exchange_config as EXCHANGE_CONFIG

# Ottieni la chiave segreta dalle variabili d'ambiente
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")

security = HTTPBearer()

class Dependencies:
    """Gestione delle dipendenze dell'applicazione"""
    _exchange: Optional[ccxt_async.Exchange] = None

    @classmethod
    async def get_exchange(cls) -> ccxt_async.Exchange:
        """
        Restituisce l'istanza dell'exchange, creandola se necessario
        """
        if cls._exchange is None:
            cls._exchange = ccxt_async.bybit(EXCHANGE_CONFIG)
            await cls._exchange.load_markets()
        return cls._exchange

    @classmethod
    async def close_exchange(cls):
        """Chiude la connessione con l'exchange"""
        if cls._exchange:
            await cls._exchange.close()
            cls._exchange = None

async def verify_auth_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Verifica il JWT Bearer e controlla che le credenziali corrispondano
    a quelle di EXCHANGE_CONFIG.
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

        if payload.get("api_key") != EXCHANGE_CONFIG["apiKey"] or \
           payload.get("api_secret") != EXCHANGE_CONFIG["secret"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenziali API non valide"
            )
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token scaduto"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token non valido"
        )

async def get_exchange_dependency():
    """
    Dependency per ottenere l'exchange
    """
    return await Dependencies.get_exchange()

def generate_auth_token(api_key, api_secret):
    """
    Genera un token JWT con le credenziali API
    """
    try:
        payload = {
            "api_key": api_key,
            "api_secret": api_secret,
            "exp": int(time.time()) + 86400  # Scade dopo 24 ore
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    except Exception as e:
        print(f"Errore nella generazione del token: {e}")
        raise
