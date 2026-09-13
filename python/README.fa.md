<div lang="fa" dir="rtl" align="right">

# Veridist

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

**از داده‌های طول عمر، به شناختی روشن‌تر از رفتار و قابلیت اطمینان برسید.**

<h2 dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> چیست؟</h2>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> کتابخانه‌ای پایتونی برای تحلیل داده‌های طول عمر و قابلیت اطمینان است. مشاهدات خرابی و بقا را به برآوردهایی تبدیل می‌کند که فرض‌ها، شمار مشاهده‌ها و اطلاعات اجرای آن‌ها قابل بررسی‌اند.</p>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> ترکیبی از <bdi dir="ltr"><em>verified</em></bdi> و <bdi dir="ltr"><em>distribution</em></bdi> است: «برازش توزیع با قابلیت راستی‌آزمایی». اجرای موفقِ محاسبه به‌معنای درست‌بودن مدل نیست؛ نتیجه باید با فرض‌هایش خوانده شود.</p>

اکنون می‌توانید:

- مدل‌های عمر نمایی، وایبول و لگ‌نرمال را برازش دهید؛
- مشاهده‌های **سانسورشده از راست**<sup id="fnref-right-censoring"><a href="#fn-right-censoring">۳</a></sup> را در تحلیل نگه دارید؛
- برآورد، شمار مشاهده‌ها و فرض‌های محاسبه را بررسی کنید؛ و
- محاسبهٔ درست‌نمایی<sup id="fnref-likelihood"><a href="#fn-likelihood">۲</a></sup> را در مسیرهای پشتیبانی‌شده به‌صورت مرحله‌ای و محلی ادامه دهید.

مسیر فایل فعلی عمداً محدود است: CSV سخت‌گیرانهٔ UTF-8 فقط مدل نماییِ نرخ‌محور را برازش می‌دهد. برای وایبول و لگ‌نرمال از شیءهای دادهٔ طول عمر استفاده می‌شود. [ماتریس قابلیت‌ها](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) مرز دقیق نسخه را ثبت می‌کند.

<h2 dir="rtl" align="right">با یک پرسش طول عمر شروع کنید</h2>

فرض کنید عمر تعدادی پمپ را بررسی می‌کنید. برای هر پمپ می‌دانید چه مدت زیر نظر بوده و آیا خراب شده است.

| وضعیت پمپ | چیزی که می‌دانیم |
| --- | --- |
| در طول مطالعه خراب شده است | زمان خرابی مشخص است. |
| در پایان مطالعه هنوز کار می‌کند | عمر واقعی آن از زمان مشاهده‌شده بیشتر است. |

نوع دوم **مشاهدهٔ سانسورشده از راست**<sup id="fnref-right-censoring"><a href="#fn-right-censoring">۳</a></sup> است: مطالعه پیش از دیدن خرابی پایان یافته است. این مشاهدات هم اطلاعات دارند.

مدل آماری توصیفی ساده‌شده از الگوی زمان‌هاست. Veridist مدل را برازش می‌دهد؛ بررسی سازگاری فرض‌های آن با رفتار واقعی سیستم بخشی از تحلیل شماست.

<h2 dir="rtl" align="right">اولین تحلیل شما</h2>

با یک مدل نمایی شروع می‌کنیم که نرخ خرابی را در طول زمان ثابت فرض می‌کند؛ مثال ساده‌ای برای یادگیری است، اما لزوماً برای تجهیزات فرسوده‌شونده مناسب نیست.

### نصب

به Python 3.11 تا 3.14 نیاز دارید:

</div>

```console
python -m pip install veridist
```

<div lang="fa" dir="rtl" align="right">

### شناخت داده

| زمان | وقوع رویداد | معنی |
| --- | --- | --- |
| ۱ | ۱ | خرابی در زمان ۱ مشاهده شده است. |
| ۱ | ۰ | پمپ تا زمان ۱ سالم مانده است. |

یک واحد زمان مانند ساعت یا ماه انتخاب کنید و در همهٔ ردیف‌ها به کار ببرید. این دو مشاهده برای توضیح محاسبه‌اند و شواهد کافی برای تصمیم واقعی دربارهٔ قابلیت اطمینان نیستند.

### برازش مدل

کد زیر فایل نمونه را خودش ایجاد می‌کند و پس از نصب قابل اجراست.

</div>

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

<div lang="fa" dir="rtl" align="right">

### تفسیر نتیجه

یک خرابی در مجموع دو واحد زمان مشاهده داریم؛ نرخ برآوردشده برابر ۰٫۵ است. اگر زمان‌ها برحسب ماه باشند، این مقدار ۰٫۵ خرابی به‌ازای هر پمپ‌ماه مشاهده است. این عدد احتمال خرابی ۵۰٪ در یک ماه نیست؛ نرخ و احتمال متفاوت‌اند.

پمپ سالم هم یک واحد زمان بدون خرابی به اطلاعات اضافه کرده است. تحلیل فرض می‌کند پایان مشاهده مستقل از زمان خرابی پنهان پمپ است؛ اگر پمپ‌های مشکوک به خرابی زودتر از مطالعه خارج شده باشند، این فرض نیاز به بررسی دارد.

اجرای موفق، درستی انتخاب مدل را اثبات نمی‌کند. [آموزش برازش نمایی و سانسورشدگی](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md) را برای تفسیر دقیق‌تر دنبال کنید.

<h2 dir="rtl" align="right">چرا توزیع احتمال را مدل کنیم؟</h2>

داده‌ها الگو، پراکندگی و رخدادهای نادر دارند. **برازش**<sup id="fnref-distribution-fitting"><a href="#fn-distribution-fitting">۱</a></sup> یک توزیع احتمال، توصیفی فشرده از این رفتار می‌دهد، محاسبهٔ احتمال را ممکن می‌کند و عدم‌قطعیت را آشکار نگه می‌دارد.

وقتی دادهٔ کافی برای آموزش و ارزیابی قابل‌اتکای مدل‌های پیچیده، مانند شبکه‌های عصبی عمیق، نداریم، مدل‌های آماری با پارامترهای کمتر می‌توانند مناسب باشند؛ اگر فرض‌هایشان با مسئله سازگار باشد. حجم داده تنها معیار انتخاب روش نیست: هدف تحلیل، ساختار داده و نیاز به توضیح‌پذیری نیز اهمیت دارند. مدل‌کردن توزیع روی داده‌های بزرگ هم کاربرد دارد.

توزیع مرجع می‌تواند برای **تشخیص ناهنجاری**<sup id="fnref-anomaly-detection"><a href="#fn-anomaly-detection">۴</a></sup> و **پایش تغییر توزیع داده‌ها**<sup id="fnref-distribution-drift"><a href="#fn-distribution-drift">۵</a></sup> مفید باشد؛ اما این کاربردها به اعتبارسنجی، آستانهٔ تصمیم و کنترل هشدارهای کاذب نیاز دارند. پارامترها، صدک‌ها و احتمال عبور از آستانه نیز می‌توانند بعدها ویژگیِ مدل‌های یادگیری عمیق باشند، به شرطی که بدون نشت اطلاعات آینده یا مجموعهٔ آزمون برآورد شوند.

<h2 dir="rtl" align="right">وقتی توزیع داده را نمی‌دانیم</h2>

برازش توزیع می‌تواند چند مدل نامزد را برازش دهد، پارامترهایشان را برآورد کند و سازگاری آن‌ها با داده را مقایسه کند. بهترین گزینهٔ رتبه‌بندی‌شده الزاماً توزیع واقعی داده نیست و ممکن است هیچ نامزدی کافی نباشد.

رتبه‌بندی خودکار بین خانواده‌های فعلی Veridist از برنامه‌های آینده است. استنباط و انتخابِ مبتنی بر کفایت فعلی دامنهٔ محدودتری دارند: نمونه‌های نماییِ مثبت، متناهی و بدون سانسور. [ماتریس قابلیت‌ها](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) قرارداد دقیق را ثبت می‌کند.

<h2 dir="rtl" align="right">با دادهٔ خودتان کار کنید</h2>

مسیر فایل CSV خود را به تابع بدهید. ورودی CSV فعلی فقط مسیر نمایی را ارائه می‌کند و باید فایل طول عمر با قالب سخت‌گیرانهٔ UTF-8 باشد.

| تنظیم | کاربرد |
| --- | --- |
| <bdi dir="ltr">CsvLifetimeSchema</bdi> | نام ستون زمان و ستون وقوع رویداد |
| <bdi dir="ltr">PublicSourceId</bdi> | شناسهٔ عمومی و غیرمحرمانهٔ منبع در اطلاعات اجرا |
| <bdi dir="ltr">CsvLifetimeLimits</bdi> | محدودیت‌های اندازهٔ ورودی برحسب بایت |

[مرجع API](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) ورودی، نوع نتیجه و خطاها را توضیح می‌دهد.

<h2 dir="rtl" align="right">مدل‌ها و ابزارهای در دسترس</h2>

### برازش داده‌های طول عمر

| مدل | رفتار قابل توصیف |
| --- | --- |
| نمایی | نرخ خرابی ثابت |
| وایبول | نرخ خرابی کاهشی، ثابت یا افزایشی، بسته به پارامتر شکل |
| لگ‌نرمال | زمان‌های مثبت که لگاریتم آن‌ها با توزیع نرمال مدل می‌شود |

برازش‌ها با پارامتر مکان ثابت صفر، برای مشاهدات دقیق و سانسورشدهٔ مستقل از راست پشتیبانی می‌شوند. وایبول و لگ‌نرمال از APIهای مدل استفاده می‌کنند؛ تابع CSV مثال فقط نمایی است.

### محاسبات توزیع احتمال

ابزارهای اسکالر نرمال، گاما، وایبول، لگ‌نرمال و گامبل راست، لگاریتم چگالی، تابع توزیع تجمعی، تابع بقا، صدک و نمونه‌گیری با مولد تصادفی تحت کنترل کاربر را فراهم می‌کنند. وجود ابزار محاسباتی یک توزیع به معنی پشتیبانی از برازش آن نیست. [راهنمای توزیع‌ها و درست‌نمایی](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/families-log-density-likelihood.md) را ببینید.

### ارزیابی مدل
<details>
<summary>نام ابزارها و جزئیات محاسبه برای برنامه‌نویسان</summary>

<bdi dir="ltr">FAMILY_REGISTRY</bdi> خانواده‌های <bdi dir="ltr">normal, gamma, weibull_min, lognormal, gumbel_right</bdi> را معرفی می‌کند. <bdi dir="ltr">evaluate_log_density</bdi> برای محاسبهٔ اسکالر است و <bdi dir="ltr">reduce_log_likelihood_chunks</bdi> جمله‌های چگالیِ لگاریتمی binary64 موفق را با مجموع صحیحِ دقیق و یک گردکردن نهایی کاهش می‌دهد.

</details>


برای نمونه‌های مثبت، متناهی و بدون سانسور در مسیر نمایی، آزمون‌های Monte Carlo با برازش مجدد KS/AD/CvM، معیارهای AIC/BIC و انتخاب مشروط به کفایت مدل با مولد تصادفی کاربر وجود دارند. دامنهٔ استنباط محدودتر از برازش است؛ [ماتریس قابلیت‌ها](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) جزئیات را ثبت می‌کند.

<h2 dir="rtl" align="right">وقتی داده بیشتر می‌شود</h2>

ابزارهای درست‌نمایی روی بخش‌های داده محاسبه می‌کنند؛ برنامهٔ شما تقسیم و تحویل بخش‌ها را مدیریت می‌کند.

<bdi dir="ltr">SQLiteCheckpointStore</bdi> وضعیت محلی را برای کاهش‌های نمایی سازگار، از جمله مسیر CSV، نگه می‌دارد. نسخهٔ منبع باید ثابت بماند. [مثال ذخیرهٔ وضعیت و ادامهٔ اجرا](https://github.com/alisadeghiaghili/veridist/blob/main/python/examples/checkpoint_resume.py) را دنبال کنید.

شواهد انتشار مسیرهای مشخص CSV و نمایی را روی ۱۰ هزار، ۱۰۰ هزار و یک میلیون ردیف در شرایط ثبت‌شده بررسی کرده‌اند؛ تضمین عمومی سرعت یا سقف مصرف حافظهٔ فرایند نیستند. بازیابی فعلی روی یک ماشین و فایل‌سیستم محلی اجرا می‌شود.

<h2 dir="rtl" align="right">کیفیت چگونه بررسی می‌شود؟</h2>

| بررسی | نقش آن |
| --- | --- |
| تست‌های مرجع آماری و قرارداد API | بررسی نتایج، شرایط مرزی و رفتار خطاها |
| حداقل پوشش ۹۵٪ | الزام پوشش خطوط و شاخه‌ها؛ ماژول‌های عددی آستانهٔ سخت‌گیرانه‌تری دارند |
| تست جهش هستهٔ آماری | سنجش تشخیص تغییرات معیوب کد توسط تست‌ها |
| ساخت و نصب بسته | بررسی ساخت و نصب خروجی انتشار |
| مثال‌ها و مستندات چندزبانه | بررسی اجرای مثال‌ها و ساخت مستندات |

بج <bdi dir="ltr">Coverage ≥95%</bdi> حداقل الزام پروژه را نشان می‌دهد، نه درصد اندازه‌گیری‌شدهٔ آخرین اجرا. بج CI وضعیت گردش‌کار اصلی است.

<h2 dir="rtl" align="right">پیش از استفاده در پروژهٔ واقعی</h2>

سازگاری فرض‌ها با جمع‌آوری داده، وجود مسیر برازش و استنباط موردنیاز و تناسب امکانات پردازش و بازیابی محلی با کارتان را بررسی کنید.

[محدودیت‌های شناخته‌شده](https://github.com/alisadeghiaghili/veridist/blob/main/python/KNOWN_LIMITS.fa.md) مواردی مانند سانسورشدگی چپ و بازه‌ای، متغیرهای توضیحی و اجرای توزیع‌شده را توضیح می‌دهد. وضعیت کد تاریخی distfit_pro در [راهنمای مهاجرت](https://github.com/alisadeghiaghili/veridist/blob/main/docs/migration/README.md) ثبت می‌شود و به معنای سازگاری اجرایی نسخهٔ فعلی نیست.

<h2 dir="rtl" align="right">برنامه‌های آینده</h2>

مسیر توسعه بر گسترش مدل‌ها، ساده‌ترکردن تحلیل و حفظ دقت آماری متمرکز است:

- **انتقال تدریجی مجموعهٔ ۲۵ توزیع قدیمی:** بررسی و انتقال ۲۰ توزیع پیوسته و ۵ توزیع گسسته، همراه با تست‌های عددی، مستندات و دامنهٔ پشتیبانی مشخص.
- **برازش و مقایسهٔ چندمدلی:** ارائهٔ نامزدهای رتبه‌بندی‌شده با پارامترها، معیارهای مقایسه و وضعیت کفایت؛ با امکان اعلام اینکه هیچ گزینه‌ای مناسب نیست.
- **گسترش ارزیابی آماری:** توسعهٔ سنجش برازش و عدم‌قطعیت برای خانواده‌ها و شرایط داده‌ای بیشتر.
- **پردازش کارآمدتر داده‌های بزرگ:** ارزیابی و بهینه‌سازی زمان اجرا و حافظه با آزمایش‌های قابل‌بازتولید، همراه با توسعهٔ پردازش مرحله‌ای.
- **آموزش‌های کاربردی بیشتر:** vignetteهایی از شناخت مسئله و داده تا تفسیر نتیجه؛ سپس نمونه‌های ویژگی‌های توزیعی برای تشخیص ناهنجاری، پایش drift و یادگیری ماشین.

این موارد جهت توسعه‌اند، نه قابلیت فعلی یا تعهد به تاریخ انتشار. امکانات منتشرشده در [ماتریس قابلیت‌ها](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) و تغییرات تحویل‌شده در [تاریخچهٔ تغییرات](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md) ثبت می‌شوند.

<h2 dir="rtl" align="right">راهنما و مشارکت</h2>

| هدف | مسیر |
| --- | --- |
| راهنمای مستقل بسته | [README بسته](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) |
| یادگیری مثال سانسورشدگی | [آموزش نمایی](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md) |
| بررسی ورودی و خروجی | [مرجع API](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) |
| گزارش مشکل قابل‌بازتولید | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| مشارکت | [راهنمای مشارکت](https://github.com/alisadeghiaghili/veridist/blob/main/CONTRIBUTING.md) و [قراردادهای مهندسی](https://github.com/alisadeghiaghili/veridist/blob/main/docs/conventions.md) |
| گزارش آسیب‌پذیری | [سیاست امنیت](https://github.com/alisadeghiaghili/veridist/blob/main/SECURITY.md) |
| پیگیری نسخه‌ها | [تاریخچهٔ تغییرات](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md) |

<h2 dir="rtl" align="right">استناد به <bdi dir="ltr">Veridist</bdi></h2>

به همان نسخه‌ای استناد کنید که نتیجه با آن تولید شده است. [راهنمای استناد](https://github.com/alisadeghiaghili/veridist/blob/main/docs/citing-veridist.md) قالب‌های IEEE، APA 7، Chicago، MLA 9، Harvard، Vancouver، BibTeX، RIS، EndNote XML و CSL-JSON را ارائه می‌کند. [CITATION.cff](https://github.com/alisadeghiaghili/veridist/blob/main/CITATION.cff) مرجع ماشین‌خوان است.

<h2 dir="rtl" align="right">نویسنده و پروفایل‌های پژوهشی</h2>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> را [سید علی صادقی عقیلی](https://zil.ink/thedatascientist) نگهداری می‌کند.</p>

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili)
[![PeerJ](https://img.shields.io/badge/PeerJ-00A4A6?logo=peerj&logoColor=white)](https://peerj.com/AliSadeghiAghili/)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-5938-3291)

<h2 dir="rtl" align="right">یادداشت اصطلاحات</h2>

در نخستین کاربرد هر اصطلاح، اندیس بالانویس به تعریف آن و معادل انگلیسی ارجاع می‌دهد.

<p id="fn-distribution-fitting" dir="rtl" align="right"><strong>۱.</strong> <bdi dir="ltr">Distribution Fitting</bdi> — انتخاب یک یا چند توزیع نامزد، برآورد پارامترهای آن‌ها و سنجش سازگاری‌شان با داده. <a href="#fnref-distribution-fitting">↩</a></p>
<p id="fn-likelihood" dir="rtl" align="right"><strong>۲.</strong> <bdi dir="ltr">Likelihood</bdi> — معیاری برای سنجش سازگاری یک مدل و پارامترهایش با مشاهدات ثبت‌شده. <a href="#fnref-likelihood">↩</a></p>
<p id="fn-right-censoring" dir="rtl" align="right"><strong>۳.</strong> <bdi dir="ltr">Right Censoring</bdi> — مشاهده‌ای که در آن زمان وقوع رویداد تا پایان بازهٔ مشاهده دیده نشده است. <a href="#fnref-right-censoring">↩</a></p>
<p id="fn-anomaly-detection" dir="rtl" align="right"><strong>۴.</strong> <bdi dir="ltr">Anomaly Detection</bdi> — شناسایی مشاهده‌هایی که با الگوی مرجع داده سازگاری کمی دارند. <a href="#fnref-anomaly-detection">↩</a></p>
<p id="fn-distribution-drift" dir="rtl" align="right"><strong>۵.</strong> <bdi dir="ltr">Distribution Drift</bdi> — تغییر معنادار در توزیع دادهٔ جدید نسبت به توزیع مرجع یا دادهٔ گذشته. <a href="#fnref-distribution-drift">↩</a></p>

<h2 dir="rtl" align="right">مجوز</h2>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> تحت <strong><bdi dir="ltr">Business Source License 1.1 — BUSL-1.1</bdi></strong> عرضه می‌شود. <a href="https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE"><bdi dir="ltr">LICENSE</bdi></a> اعطای استفادهٔ اضافی تحت <bdi dir="ltr">Apache-2.0</bdi> با شرایط مشخص و تاریخ تغییر مجوز را تعیین می‌کند. بج <bdi dir="ltr">BUSL-1.1</bdi> به معنای عرضهٔ فعلی بدون قیدوشرط تحت <bdi dir="ltr">Apache-2.0</bdi> نیست.</p>

</div>
