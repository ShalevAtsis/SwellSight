"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/providers/AuthProvider";

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { token, ready } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (ready && !token) router.replace("/login");
  }, [ready, token, router]);

  if (!ready) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center text-foam-500">
        Loading…
      </div>
    );
  }
  if (!token) return null;
  return <>{children}</>;
}
