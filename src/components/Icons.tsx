/** The app's icons: one stroke style, sized by the `size` prop, coloured by the text colour. */
import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function icon(body: ReactNode, filled = false) {
  return function Icon({ size = 16, ...rest }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill={filled ? "currentColor" : "none"}
        stroke={filled ? "none" : "currentColor"}
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
        {...rest}
      >
        {body}
      </svg>
    );
  };
}

export const IconPlay = icon(<path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5Z" />, true);
export const IconPause = icon(<><rect x="6" y="4" width="4.5" height="16" rx="1.2" /><rect x="13.5" y="4" width="4.5" height="16" rx="1.2" /></>, true);
export const IconPrevRally = icon(<><path d="M19 5v14l-10-7 10-7Z" /><path d="M5 5v14" /></>);
export const IconNextRally = icon(<><path d="M5 5v14l10-7-10-7Z" /><path d="M19 5v14" /></>);
export const IconStepBack = icon(<path d="m15 18-6-6 6-6" />);
export const IconStepForward = icon(<path d="m9 18 6-6-6-6" />);
export const IconRecord = icon(<><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="3.5" fill="currentColor" /></>);
export const IconUpload = icon(<><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>);
export const IconFolder = icon(<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />);
export const IconTrash = icon(<><path d="M4 7h16" /><path d="M9 7V4.5h6V7" /><path d="m6 7 1 13h10l1-13" /></>);
export const IconCheck = icon(<path d="m5 12.5 4.5 4.5L19 7.5" />);
export const IconClose = icon(<path d="M6 6l12 12M18 6 6 18" />);
export const IconAlert = icon(<><path d="M12 4 2.8 19.5a.5.5 0 0 0 .43.75h17.54a.5.5 0 0 0 .43-.75Z" /><path d="M12 10v4.5" /><path d="M12 17.5v.01" /></>);
export const IconRefresh = icon(<><path d="M4 12a8 8 0 0 1 13.7-5.6L20 8.5" /><path d="M20 4v4.5h-4.5" /><path d="M20 12a8 8 0 0 1-13.7 5.6L4 15.5" /><path d="M4 20v-4.5h4.5" /></>);
export const IconSplit = icon(<><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M12 5v14" /></>);
export const IconLayers = icon(<><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>);
export const IconEdit = icon(<><path d="M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17Z" /><path d="m14 8 3 3" /></>);
export const IconTarget = icon(<><path d="M4 8V5a1 1 0 0 1 1-1h3" /><path d="M16 4h3a1 1 0 0 1 1 1v3" /><path d="M20 16v3a1 1 0 0 1-1 1h-3" /><path d="M8 20H5a1 1 0 0 1-1-1v-3" /><path d="M8 15.5 9.5 8.5h5l1.5 7Z" /></>);
export const IconServer = icon(<><rect x="4" y="4" width="16" height="7" rx="1.5" /><rect x="4" y="13" width="16" height="7" rx="1.5" /><path d="M8 7.5v.01M8 16.5v.01" /></>);
export const IconArrowRight = icon(<><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>);
export const IconFilm = icon(<><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M8 5v14M16 5v14M3 12h18" /></>);

/** The product mark: a ball's seams, in the accent colour. */
export function BrandMark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <circle cx="16" cy="16" r="14" fill="var(--accent)" />
      <path
        d="M16 2c1.2 6.4-.6 11.2-5.4 14.6M16 16c4.6-3.7 9.800-4.300 14-1.800M16 16c-.9 5.900 1.200 10.700 6.500 14.200M10.600 16.600C7.300 18.900 4.700 19.500 2.300 19M16 16c-4.200 2.600-7.200 6.600-8 11.200M16 2c5 2.200 8.800 6.800 9.700 12"
        stroke="var(--accent-ink)"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}
