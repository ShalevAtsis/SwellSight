"use client";

import Link from "next/link";
import { useAuth } from "@/components/providers/AuthProvider";

export function Header() {
  const { token, ready, logout } = useAuth();

  return (
    <header className="border-b border-ocean-800/40 bg-ocean-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link href="/" className="font-display text-xl font-semibold text-foam-100 tracking-tight">
          Swell<span className="text-swell-400">Sight</span>
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          {ready && token ? (
            <>
              <Link href="/analyze" className="text-foam-200 hover:text-foam-50 transition">
                Analyze
              </Link>
              <Link href="/history" className="text-foam-200 hover:text-foam-50 transition">
                History
              </Link>
              <button
                type="button"
                onClick={logout}
                className="rounded-lg border border-ocean-700 px-3 py-1.5 text-foam-300 hover:bg-ocean-800 transition"
              >
                Log out
              </button>
            </>
          ) : ready ? (
            <>
              <Link href="/login" className="text-foam-200 hover:text-foam-50 transition">
                Log in
              </Link>
              <Link
                href="/register"
                className="rounded-lg bg-swell-500 px-3 py-1.5 font-medium text-ocean-950 hover:bg-swell-400 transition"
              >
                Sign up
              </Link>
            </>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
