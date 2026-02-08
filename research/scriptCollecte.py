# -*- coding: utf-8 -*-
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from github import Github, Auth
from github.GithubException import RateLimitExceededException, GithubException
import itertools
# ---------------- CONFIG ----------------
TOKENS = [
    
]
SAVE_FILE = "resultats5200000-5500000-100repos.txt"
MAX_USERS = 300000
SAFE_MARGIN = 100
MIN_STARRED_REPOS = 100
MIN_REPO_STARS = 5000
MIN_RESULTS_PER_USER = 100
DEBUT = 5200000
N_THREADS = min(32, max(1, len(TOKENS)))
RATE_CHECK_COOLDOWN = 1.5
# ----------------------------------------

# ---------- Compteurs globaux ----------
USERS_PROCESSED = 0       # 👇 nombre total d’utilisateurs traités
USERS_SELECTED = 0        # 👇 nombre total d’utilisateurs valides
COUNTER_LOCK = threading.Lock()
# ---------------------------------------

# ---------- Gestion des tokens ----------
class TokenClient:
    def __init__(self, token: str, index: int):
        self.index = index
        self.token = token
        self.client = Github(auth=Auth.Token(token), per_page=100)
        self._last_checked = 0.0
        self._rate = None
        self.lock = threading.Lock()

    def refresh_rate(self):
        now = time.time()
        if now - self._last_checked < RATE_CHECK_COOLDOWN and self._rate is not None:
            return self._rate
        try:
            rate = self.client.get_rate_limit().rate
            remaining = getattr(rate, "remaining", 0)
            limit = getattr(rate, "limit", 0)
            reset = getattr(rate, "reset", datetime.now(timezone.utc))
            self._rate = (remaining, limit, reset)
        except Exception:
            self._rate = (0, 0, datetime.now(timezone.utc))
        self._last_checked = now
        return self._rate

class TokenPool:
    def __init__(self, tokens: List[str], safe_margin: int = SAFE_MARGIN):
        self.clients = [TokenClient(t, i) for i, t in enumerate(tokens)]
        self.safe_margin = safe_margin

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> Tuple[TokenClient, int]:
        start = time.time()
        while True:
            for tc in self.clients:
                rem, limit, reset = tc.refresh_rate()
                if rem > self.safe_margin:
                    if tc.lock.acquire(blocking=False):
                        return tc, tc.index
            if timeout is not None and (time.time() - start) >= timeout:
                raise TimeoutError("Aucun token disponible avant expiration du timeout.")
            time.sleep(1.0)

    def release(self, tc: TokenClient):
        try:
            tc.lock.release()
        except RuntimeError:
            pass

# ---------- Fonctions utilitaires ----------
def save_results(data: List[Tuple[str, List[str]]]):
    if not data:
        return
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        for login, repos in data:
            f.write(f"{login} : [{', '.join(repos)}]\n")
    print(f"💾 Sauvegardé {len(data)} utilisateurs dans {SAVE_FILE}")

def monitor_tokens(pool: TokenPool, interval: int = 60):
    """Affiche le quota restant + progression utilisateur toutes les 60 s."""
    global USERS_PROCESSED, USERS_SELECTED

    # ✅ Récupère une seule fois le login associé à chaque token
    token_logins = []
    for tc in pool.clients:
        try:
            login = tc.client.get_user().login
        except Exception:
            login = "❌ Inconnu"
        token_logins.append(login)

    print("\n🔐 Vérification des tokens au démarrage :")
    for i, login in enumerate(token_logins):
        print(f"  🪙 Token {i} → Compte : {login}")
    print("------------------------\n")

    # ✅ Boucle de monitoring continue
    while True:
        now = datetime.now(timezone.utc)
        print("\n📊 --- État actuel ---")
        for i, tc in enumerate(pool.clients):
            rem, limit, reset = tc.refresh_rate()
            wait_min = max((reset - now).total_seconds() / 60.0, 0)
            print(f"🔎 Token {i} ({token_logins[i]}): {rem}/{limit} restantes, reset dans {wait_min:.1f} min")

        with COUNTER_LOCK:
            print(f"👤 Utilisateurs traités : {USERS_PROCESSED}/{MAX_USERS}")
            print(f"⭐ Utilisateurs sélectionnés : {USERS_SELECTED}")
        print("------------------------\n")

        time.sleep(interval)

# ---------- Traitement utilisateur ----------
def process_user(pool: TokenPool, user_login: str) -> Optional[Tuple[str, List[str]]]:
    global USERS_PROCESSED, USERS_SELECTED

    try:
        tc, idx = pool.acquire(timeout=30)
    except TimeoutError:
        print(f"⏱️ Pas de token dispo pour {user_login}, replanification.")
        return "RETRY"

    client = tc.client
    try:
        gh_user = client.get_user(user_login)
        starred = gh_user.get_starred()
        total = getattr(starred, "totalCount", None)
        if total is not None and total < MIN_STARRED_REPOS:
            with COUNTER_LOCK:
                USERS_PROCESSED += 1
            return None

        found = []
        for repo in starred:
            stars = repo.raw_data.get("stargazers_count", 0)
            if stars >= MIN_REPO_STARS:
                found.append(repo.full_name)
                if len(found) >= MIN_RESULTS_PER_USER:
                    break

        with COUNTER_LOCK:
            USERS_PROCESSED += 1
            if len(found) >= MIN_RESULTS_PER_USER:
                USERS_SELECTED += 1

        if len(found) >= MIN_RESULTS_PER_USER:
            return (user_login, found)
        return None

    except RateLimitExceededException:
        print(f"🚫 Token {idx} épuisé pendant {user_login}. Replanification.")
        return "RETRY"
    except GithubException as e:
        print(f"⚠️ Erreur Github {user_login}: {e.status} - {e.data}")
        return None
    except Exception as e:
        print(f"⚠️ Erreur inattendue {user_login}: {e}")
        return None
    finally:
        pool.release(tc)

# ---------- Main ----------
def main():
    if not TOKENS:
        raise SystemExit("❌ Aucun token défini dans TOKENS.")

    pool = TokenPool(TOKENS, safe_margin=SAFE_MARGIN)
    threading.Thread(target=monitor_tokens, args=(pool, 60), daemon=True).start()

    starter_client = Github(auth=Auth.Token(TOKENS[0]), per_page=100)
    users_iter = starter_client.get_users(since=0)
    subset_logins = [u.login for _, u in zip(range(MAX_USERS), itertools.islice(users_iter, DEBUT, None))]

    print(f"🚀 Analyse parallèle sur {len(subset_logins)} utilisateurs ({N_THREADS} threads)")
    results_accum = []
    retry_queue = []

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(process_user, pool, login): login for login in subset_logins}

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                login = futures.pop(fut)
                try:
                    res = fut.result()
                    if res == "RETRY":
                        retry_queue.append(login)
                    elif res:
                        results_accum.append(res)
                        if len(results_accum) >= 10:
                            save_results(results_accum)
                            results_accum = []
                except Exception as e:
                    print(f"❌ Erreur tâche {login}: {e}")

            if retry_queue:
                time.sleep(10)
                for login in retry_queue[:]:
                    futures[ex.submit(process_user, pool, login)] = login
                retry_queue.clear()

    if results_accum:
        save_results(results_accum)

    print("✅ Analyse terminée.")

if __name__ == "__main__":
    main()
