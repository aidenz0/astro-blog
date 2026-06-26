import type { CollectionEntry } from "astro:content";
import postFilter from "./postFilter";
import { slugifyStr } from "./slugify";

/**
 * 取某系列下的全部文章，按 seriesOrder 升序排列
 * （缺失 seriesOrder 时排到末尾，并按发布时间升序兜底）。
 * 与 getSortedPosts 的日期倒序不同：系列要从第 1 篇顺读到最后一篇。
 */
const getPostsBySeries = (
  posts: CollectionEntry<"blog">[],
  series: string
): CollectionEntry<"blog">[] =>
  posts
    .filter(postFilter)
    .filter(
      post => post.data.series && slugifyStr(post.data.series) === series
    )
    .sort((a, b) => {
      const orderA = a.data.seriesOrder ?? Infinity;
      const orderB = b.data.seriesOrder ?? Infinity;
      if (orderA !== orderB) return orderA - orderB;
      return (
        new Date(a.data.pubDatetime).getTime() -
        new Date(b.data.pubDatetime).getTime()
      );
    });

export default getPostsBySeries;
