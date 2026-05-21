import Link from "next/link";

export default function HomePage() {
  return (
    <div className="hero-gradient">
      <section className="mx-auto max-w-5xl px-4 py-20 text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-swell-400">
          AI beach-cam analysis
        </p>
        <h1 className="mt-4 font-display text-4xl font-bold text-foam-50 sm:text-5xl text-balance">
          Know the surf before you paddle out
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-foam-400 text-balance">
          Upload a snapshot from your local cam. SwellSight estimates wave height,
          direction, breaking type, and a 0–100 surf score in seconds.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/register"
            className="rounded-xl bg-swell-500 px-6 py-3 font-semibold text-ocean-950 hover:bg-swell-400 transition shadow-lg shadow-swell-500/20"
          >
            Get started free
          </Link>
          <Link
            href="/analyze"
            className="rounded-xl border border-ocean-700 px-6 py-3 font-medium text-foam-200 hover:bg-ocean-900 transition"
          >
            Analyze a photo
          </Link>
        </div>
        <ul className="mx-auto mt-16 grid max-w-3xl gap-6 text-left sm:grid-cols-3">
          {[
            { title: "Height & direction", body: "Meters and feet with model confidence." },
            { title: "Breaking type", body: "Spilling, plunging, or surging detection." },
            { title: "Surf score", body: "0–100 score with a transparent breakdown." },
          ].map((item) => (
            <li
              key={item.title}
              className="rounded-xl border border-ocean-800 bg-ocean-900/40 p-5"
            >
              <h2 className="font-semibold text-foam-100">{item.title}</h2>
              <p className="mt-2 text-sm text-foam-500">{item.body}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
