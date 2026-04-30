/**
 * Calculate reading time and word count for a blog post
 * @param content - The post content (HTML or Markdown)
 * @param wordsPerMinute - Average reading speed (default: 300 for Chinese, 200 for English)
 * @returns Reading time in minutes and word count
 */
export function getReadingTime(
  content: string,
  wordsPerMinute: number = 300
): { minutes: number; words: number; text: string } {
  // Remove HTML tags and get plain text
  const plainText = content
    .replace(/<[^>]*>/g, '') // Remove HTML tags
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();

  // Count words (works for both Chinese and English)
  // For Chinese: count characters (excluding punctuation and spaces)
  // For English: count words separated by spaces
  const chineseChars = plainText.replace(/[\s\w\d_p{P}]+/gu, '').length;
  const englishWords = plainText.match(/[a-zA-Z]+/g)?.length || 0;

  // Total word count (Chinese characters + English words)
  const totalWords = chineseChars + englishWords;

  // Calculate reading time
  const minutes = Math.max(1, Math.ceil(totalWords / wordsPerMinute));

  // Format text
  const text = `${minutes} 分钟阅读 · ${totalWords} 字`;

  return {
    minutes,
    words: totalWords,
    text,
  };
}
