"use client";

import { FormEvent, useState } from "react";
import { ApiClientError } from "@/lib/api";
import { ErrorAlert } from "@/components/ui/ErrorAlert";

interface AuthFormProps {
  mode: "login" | "register";
  onSubmit: (email: string, password: string) => Promise<void>;
}

export function AuthForm({ mode, onSubmit }: AuthFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await onSubmit(email, password);
    } catch (err) {
      setError(
        err instanceof ApiClientError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Something went wrong",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && <ErrorAlert message={error} />}
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-foam-300">
          Email
        </label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mt-1 w-full rounded-lg border border-ocean-700 bg-ocean-900 px-3 py-2 text-foam-100 focus:border-swell-500 focus:outline-none focus:ring-1 focus:ring-swell-500"
        />
      </div>
      <div>
        <label htmlFor="password" className="block text-sm font-medium text-foam-300">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          required
          minLength={mode === "register" ? 8 : 1}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 w-full rounded-lg border border-ocean-700 bg-ocean-900 px-3 py-2 text-foam-100 focus:border-swell-500 focus:outline-none focus:ring-1 focus:ring-swell-500"
        />
        {mode === "register" && (
          <p className="mt-1 text-xs text-foam-600">At least 8 characters</p>
        )}
      </div>
      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-xl bg-swell-500 py-2.5 font-semibold text-ocean-950 hover:bg-swell-400 disabled:opacity-50 transition"
      >
        {loading ? "Please wait…" : mode === "login" ? "Log in" : "Create account"}
      </button>
    </form>
  );
}
