import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

interface Series {
  series: string; // slug
  seriesName: string; // 原始系列名
  count: number; // 系列内文章数
}

/**
 * 提取所有去重后的系列及其文章数（仿 getUniqueTags）。
 * 只统计设置了 series 字段、且通过 postFilter（非草稿/已发布）的文章。
 */
const getUniqueSeries = (posts: CollectionEntry<"blog">[]): Series[] => {
  const map = new Map<string, Series>();

  posts
    .filter(postFilter)
    .filter(post => post.data.series)
    .forEach(post => {
      const seriesName = post.data.series as string;
      const series = slugifyStr(seriesName);
      const existing = map.get(series);
      if (existing) {
        existing.count += 1;
      } else {
        map.set(series, { series, seriesName, count: 1 });
      }
    });

  return [...map.values()].sort((a, b) => a.series.localeCompare(b.series));
};

export default getUniqueSeries;
