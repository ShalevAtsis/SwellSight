"use client";

import Link from "next/link";

export default function ForgotPasswordPage() {
  return (
    <div className="mx-auto max-w-md px-4 py-12">
      <h1 className="font-display text-2xl font-bold text-foam-50">
        Reset password
      </h1>
      <p className="mt-4 text-sm text-foam-400 leading-relaxed">
        Email-based password reset is not enabled in this environment yet.
        Contact your administrator or create a new account if you lost access.
      </p>
      <p className="mt-6 text-sm text-foam-500">
        Planned for a future release (SMTP or auth provider integration).
      </p>
      <Link
        href="/login"
        className="mt-8 inline-block text-swell-400 hover:underline"
      >
        Back to log in
      </Link>
    </div>
  );
}
