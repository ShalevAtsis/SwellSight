"use client";

import Link from "next/link";
import { AuthForm } from "@/components/auth/AuthForm";
import { useAuth } from "@/components/providers/AuthProvider";

export default function RegisterPage() {
  const { register } = useAuth();

  return (
    <div className="mx-auto max-w-md px-4 py-12">
      <h1 className="font-display text-2xl font-bold text-foam-50">Sign up</h1>
      <p className="mt-2 text-sm text-foam-500">
        Already have an account?{" "}
        <Link href="/login" className="text-swell-400 hover:underline">
          Log in
        </Link>
      </p>
      <div className="mt-8">
        <AuthForm mode="register" onSubmit={register} />
      </div>
    </div>
  );
}
