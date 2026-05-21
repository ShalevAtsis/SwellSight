"use client";

import Link from "next/link";
import { AuthForm } from "@/components/auth/AuthForm";
import { useAuth } from "@/components/providers/AuthProvider";

export default function LoginPage() {
  const { login } = useAuth();

  return (
    <div className="mx-auto max-w-md px-4 py-12">
      <h1 className="font-display text-2xl font-bold text-foam-50">Log in</h1>
      <p className="mt-2 text-sm text-foam-500">
        New here?{" "}
        <Link href="/register" className="text-swell-400 hover:underline">
          Create an account
        </Link>
      </p>
      <div className="mt-8">
        <AuthForm mode="login" onSubmit={login} />
      </div>
    </div>
  );
}
