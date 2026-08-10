import styles from './YouTubeEmbed.module.css';

export type YouTubeEmbedProps = {
  videoId: string;
  title: string;
  caption?: string;
};

export default function YouTubeEmbed({videoId, title, caption}: YouTubeEmbedProps) {
  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        <iframe
          src={`https://www.youtube-nocookie.com/embed/${videoId}`}
          title={title}
          loading="lazy"
          referrerPolicy="strict-origin-when-cross-origin"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
        />
      </div>
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  );
}
