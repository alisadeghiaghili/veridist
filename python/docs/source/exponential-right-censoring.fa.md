<a id="veridist-exponential-right-censoring-fa"></a>
<h1 dir="rtl" align="right">آموزش داده‌های سانسورشده از راست در مدل نمایی</h1>

<p dir="rtl" align="right"><a href="exponential-right-censoring.md">انگلیسی</a> | <a href="exponential-right-censoring.fa.md">فارسی</a> | <a href="exponential-right-censoring.de.md">آلمانی</a></p>

<p dir="rtl" align="right">این آموزش اولین مسیر تحلیل دادهٔ طول عمر در <bdi dir="ltr">Veridist</bdi> را نشان می‌دهد: برازش توزیع<sup id="fnref-fitting"><a href="#fn-fitting">۱</a></sup> نمایی وقتی بخشی از مشاهده‌ها سانسور از راست<sup id="fnref-right-censoring"><a href="#fn-right-censoring">۲</a></sup> هستند. از این مسیر زمانی استفاده کنید که هر سطر داده می‌گوید «رخداد در این زمان اتفاق افتاده» یا «تا این زمان هنوز رخداد دیده نشده است».</p>

<h2 dir="rtl" align="right">ایدهٔ داده چیست؟</h2>

<p dir="rtl" align="right">هر سطر طول عمر دو فیلد دارد:</p>

<table class="docutils" dir="rtl" width="100%">
  <thead>
    <tr>
      <th width="30%" align="center"><p align="center">فیلد CSV</p></th>
      <th width="70%" align="center"><p align="center">معنا</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td dir="ltr" align="left"><code class="literal">time</code></td>
      <td align="right">مقدار متناهی و نامنفی از زمان مشاهده‌شده.</td>
    </tr>
    <tr>
      <td dir="ltr" align="left"><code class="literal">event_observed</code></td>
      <td align="right"><code class="literal">1</code> یعنی رخداد در <code class="literal">time</code> اتفاق افتاده است؛ <code class="literal">0</code> یعنی تا <code class="literal">time</code> هنوز رخداد اتفاق نیفتاده است.</td>
    </tr>
  </tbody>
</table>

<p dir="rtl" align="right">برای نمونه، سطر <bdi dir="ltr"><code>1,1</code></bdi> یعنی رخداد در زمان ۱ اتفاق افتاده است. سطر <bdi dir="ltr"><code>1,0</code></bdi> یعنی آن واحد تا زمان ۱ مشاهده شده و در آن لحظه هنوز زنده، سالم، فعال یا بدون رخداد بوده است. همین سطر دوم هم اطلاعات دارد: به مدل می‌گوید طول عمر واقعی بیشتر از ۱ است. این همان سانسور مستقل از راست است.</p>

<h3 dir="rtl" align="right">همین دو نوع سطر در حوزه‌های دیگر</h3>

<p dir="rtl" align="right">«طول عمر» در اینجا یعنی زمان تا یک رویداد تعریف‌شده. با تغییر موضوع و رویداد، فرمول محاسبه عوض نمی‌شود؛ اما معنای علمی و فرض‌های آن باید دوباره بررسی شوند.</p>

<div dir="rtl" align="right">

| حوزهٔ مثال | سطر دارای رویداد | سطرِ هنوز بدون رویداد |
| --- | --- | --- |
| مثال تولید | قطعه در ساعت ثبت‌شده خراب شد | قطعه تا پایان بازرسی هنوز کار می‌کرد |
| مثال پژوهش سلامت | عود بیماری یا بستری دوباره در روز ثبت‌شده رخ داد | تا آخرین پیگیری این رخداد دیده نشد |
| مثال اعتبار و بیمه | نکول یا نخستین خسارت در ماه ثبت‌شده رخ داد | تا پایان بررسی چنین رخدادی ثبت نشد |
| مثال محصول دیجیتال | ریزش یا تبدیل در روز ثبت‌شده رخ داد | کاربر تا پایان بازه فعال و بدون رویداد بود |
| مثال عملیات | تعمیر، تحویل یا پروندهٔ خدمت کامل شد | هنگام پایان مشاهده، پرونده هنوز باز بود |

</div>

<p dir="rtl" align="right">داده‌های تقلب و امنیت سایبری هم ممکن است زمان یک رویداد، مانند زمان تا نخستین هشدار، را داشته باشند. کاربرد متفاوت دیگر، مقایسهٔ مبلغ تراکنش یا فاصلهٔ زمانی با یک توزیع مرجع و ساختن نشانهٔ ناهنجاری است. این آموزش فقط حالت اول را پیاده می‌کند: یک زمان و یک نشانگر رخداد در هر سطر. این آموزش سامانهٔ کامل کشف تقلب نمی‌سازد.</p>

<h2 dir="rtl" align="right">نمونه را اجرا کنید</h2>

<p dir="rtl" align="right">تعریف schema در کد زیر تنها جفت سرستون پذیرفته‌شده را مشخص می‌کند و شناسه‌های ماشین‌خوان را عمداً چپ‌به‌راست نگه می‌دارد.</p>

```python
from veridist import CsvLifetimeSchema

schema = CsvLifetimeSchema("time", "event_observed")
```

<p dir="rtl" align="right">مثال کامل قابل اجرا همان schema را با یک فایل CSV کوچک به کار می‌برد.</p>

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
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

<h2 dir="rtl" align="right">نتیجه را چطور بخوانید؟</h2>

<p dir="rtl" align="right">این مثال یک رخداد مشاهده‌شده و دو واحد زمان مشاهده‌شده دارد: یک واحد از سطر رخداد قطعی، و یک واحد از سطر سانسورشده از راست. بنابراین برآورد بیشینهٔ درست‌نمایی<sup id="fnref-mle"><a href="#fn-mle">۳</a></sup> در مدل نمایی چنین است:</p>

<p dir="ltr" align="center"><strong>λ̂ = r / τ = 1 / 2 = 0.5</strong></p>

<p dir="rtl" align="right">در این فرمول، <bdi dir="ltr"><code>λ̂</code></bdi> نرخ<sup id="fnref-rate"><a href="#fn-rate">۴</a></sup> برآوردشده است. مقدار <bdi dir="ltr"><code>0.5</code></bdi> یعنی مدل برآورد می‌کند که در واحدهای مشابه، به‌طور میانگین به ازای هر واحد زمان مشاهده‌شده نیم رخداد روی می‌دهد. برای تبدیل نرخ به احتمال، باید یک بازهٔ زمانی مشخص داشته باشیم. اگر زمان با ساعت ثبت شده باشد، واحد نرخ «رخداد بر ساعت» است؛ اگر با روز ثبت شده باشد، واحد آن «رخداد بر روز» است.</p>

<h2 dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> چه فرضی می‌گیرد؟</h2>

<p dir="rtl" align="right">مدل فرض می‌کند سانسور مستقل از زمان رخداد است. ساده‌تر بگوییم، علت پایان مشاهده نباید خودش اطلاعات پنهانی دربارهٔ زمان رخداد داشته باشد. <bdi dir="ltr">Veridist</bdi> این فرض را در نتیجه و مستندات روشن می‌کند؛ اما نمی‌تواند آن را از خود فایل CSV اثبات کند.</p>

<p dir="rtl" align="right">در مسیر عمومی فعلی، فایل <bdi dir="ltr">CSV</bdi> باید با کدگذاری <bdi dir="ltr">UTF-8</bdi> و دقیقاً مطابق قالب ورودی توضیح‌داده‌شده در بالا باشد. <bdi dir="ltr">Veridist</bdi> نام ستون‌ها، جداکننده، کدگذاری، مقدارهای گمشده یا معنای صفر و یک را حدس نمی‌زند. اگر فایل با این قالب سازگار نباشد، پردازش متوقف می‌شود و خطا به‌روشنی گزارش می‌شود؛ برنامه مقدارهای مشکوک را خودکار تغییر نمی‌دهد.</p>

<h2 dir="rtl" align="right">حالت‌های شکست و محدودیت‌های فعلی</h2>

<p dir="rtl" align="right">وقتی نمونه خالی باشد، هیچ رخدادی مشاهده نشده باشد، مجموع زمان مشاهده‌شده برای یک رخداد صفر باشد، یا سرریز عددی<sup id="fnref-numeric-overflow"><a href="#fn-numeric-overflow">۵</a></sup> رخ دهد، برآورد آماری معتبر برگردانده نمی‌شود. در این حالت‌ها، خروجی نوع‌دارِ بدون برآورد<sup id="fnref-non-estimate"><a href="#fn-non-estimate">۶</a></sup> علت را مشخص می‌کند تا برنامه بتواند آن را از خطای خواندن فایل تشخیص دهد.</p>

<p dir="rtl" align="right">این مسیر امروز فاصلهٔ اطمینان ندارد و فعلاً فاصلهٔ اطمینان<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">۷</a></sup>، آزمون مناسب‌بودن مدل<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">۸</a></sup>، برش داده<sup id="fnref-truncation"><a href="#fn-truncation">۹</a></sup>، سانسور چپ<sup id="fnref-left-censoring"><a href="#fn-left-censoring">۱۰</a></sup>، سانسور فاصله‌ای<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">۱۱</a></sup>، وزن<sup id="fnref-weight"><a href="#fn-weight">۱۲</a></sup>، متغیر کمکی<sup id="fnref-covariate"><a href="#fn-covariate">۱۳</a></sup>، پارامتر مکان آزاد<sup id="fnref-free-location"><a href="#fn-free-location">۱۴</a></sup> یا انتخاب خودکار مدل ندارد.</p>

<details dir="rtl" align="right">
<summary>یادداشت‌های فنی دربارهٔ مقیاس و شواهد</summary>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> سطرهای فایل <bdi dir="ltr">CSV</bdi> را یک بار می‌خواند و آن‌ها را بخش‌به‌بخش پردازش می‌کند؛ بنابراین لازم نیست کل فایل را یکجا به‌صورت یک شیء در حافظه بارگذاری کند. محدودیت‌های بایتی فقط حجم محتوای پردازش‌شده‌ای را کنترل می‌کنند که این بخش از برنامه هم‌زمان نگه می‌دارد. این تنظیمات سقف کل حافظهٔ مصرفی پایتون یا سیستم‌عامل نیستند و سرعت مشخصی را نیز تضمین نمی‌کنند.</p>

<p dir="rtl" align="right">شاهد تاریخی <bdi dir="ltr"><code>SCALE-CSV-EXP-01</code></bdi> نتیجهٔ اجرای همین مسیر ورودی <bdi dir="ltr">CSV</bdi> و همین برآوردگر را روی ۱۰ هزار، ۱۰۰ هزار و ۱ میلیون سطر، با اندازه‌بخش‌های ۳۲، ۶۴ و ۱۲۸ <bdi dir="ltr">KiB</bdi> ثبت کرده است. اطلاعات منشأ<sup id="fnref-provenance"><a href="#fn-provenance">۱۵</a></sup> در قالب قدیمی این گزارش برای اثبات عملکرد نسخهٔ فعلی کافی نیست. هر ادعای تازه دربارهٔ عملکرد باید با اجرای جدید، نسخهٔ دقیق کد و مشخصات محیط آزمایش همراه باشد.</p>

<p dir="rtl" align="right">آن snapshot تاریخی پشتیبانی عمومی از دادهٔ بزرگ، آداپتور دیگر، لغو، تلاش دوباره یا checkpointing را اثبات نمی‌کند. برای ذخیره و ادامهٔ اجرا، راهنمای API و مثال اختصاصی checkpoint را ببینید.</p>

</details>

<h2 dir="rtl" align="right">اصطلاحات این آموزش</h2>

<p id="fn-fitting" dir="rtl" align="right"><strong>۱.</strong> <bdi dir="ltr">Distribution fitting</bdi> — برآورد پارامترهای یک توزیع از روی داده و بررسی اینکه آیا آن توزیع توصیف قابل‌دفاعی از مشاهده‌ها هست یا نه. <a href="#fnref-fitting" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-right-censoring" dir="rtl" align="right"><strong>۲.</strong> <bdi dir="ltr">Right censoring</bdi> — مشاهده‌ای که در آن رخداد تا آخرین زمان مشاهده‌شده اتفاق نیفتاده و فقط می‌دانیم طول عمر واقعی از آن زمان بیشتر است. <a href="#fnref-right-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-mle" dir="rtl" align="right"><strong>۳.</strong> <bdi dir="ltr">Maximum-likelihood estimate</bdi> — مقدار پارامتری که دادهٔ مشاهده‌شده را با مدل آماری انتخاب‌شده سازگارتر می‌کند. <a href="#fnref-mle" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-rate" dir="rtl" align="right"><strong>۴.</strong> <bdi dir="ltr">Rate</bdi> — تعداد مورد انتظار رخداد در هر واحد زمان؛ نرخ همان احتمال رخداد نیست. <a href="#fnref-rate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-numeric-overflow" dir="rtl" align="right"><strong>۵.</strong> <bdi dir="ltr">Numeric overflow</bdi> — حالتی که نتیجهٔ یک محاسبه از محدوده‌ای که قالب عددی رایانه می‌تواند به‌درستی نمایش دهد بزرگ‌تر می‌شود. <a href="#fnref-numeric-overflow" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-non-estimate" dir="rtl" align="right"><strong>۶.</strong> <bdi dir="ltr">Typed non-estimate</bdi> — خروجی ساختاریافته‌ای که می‌گوید ورودی پردازش شد، اما داده اجازهٔ برآورد آماری معتبر نداد. <a href="#fnref-non-estimate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-confidence-interval" dir="rtl" align="right"><strong>۷.</strong> <bdi dir="ltr">Confidence interval</bdi> — بازه‌ای که عدم‌قطعیت پیرامون یک برآورد را طبق یک روش آماری مشخص نشان می‌دهد. <a href="#fnref-confidence-interval" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-goodness-of-fit" dir="rtl" align="right"><strong>۸.</strong> <bdi dir="ltr">Goodness-of-fit test</bdi> — بررسی آماری برای اینکه شکل توزیع انتخاب‌شده چقدر با داده سازگار است. <a href="#fnref-goodness-of-fit" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-truncation" dir="rtl" align="right"><strong>۹.</strong> <bdi dir="ltr">Truncation</bdi> — فرایند نمونه‌گیری‌ای که در آن بخشی از جامعه اصلاً وارد نمونهٔ مشاهده‌شده نمی‌شود. <a href="#fnref-truncation" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-left-censoring" dir="rtl" align="right"><strong>۱۰.</strong> <bdi dir="ltr">Left censoring</bdi> — حالتی که فقط می‌دانیم رخداد پیش از یک زمان مشخص اتفاق افتاده است. <a href="#fnref-left-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-interval-censoring" dir="rtl" align="right"><strong>۱۱.</strong> <bdi dir="ltr">Interval censoring</bdi> — حالتی که فقط می‌دانیم رخداد بین دو زمان اتفاق افتاده است. <a href="#fnref-interval-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-weight" dir="rtl" align="right"><strong>۱۲.</strong> <bdi dir="ltr">Weight</bdi> — عددی که سهم یک مشاهده را در محاسبه بیشتر یا کمتر می‌کند. <a href="#fnref-weight" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-covariate" dir="rtl" align="right"><strong>۱۳.</strong> <bdi dir="ltr">Covariate</bdi> — متغیر اضافه‌ای مثل دما یا فشار که ممکن است به توضیح طول عمر کمک کند. <a href="#fnref-covariate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-free-location" dir="rtl" align="right"><strong>۱۴.</strong> <bdi dir="ltr">Free location parameter</bdi> — پارامتر جابه‌جایی که به‌جای ثابت‌بودن، از داده برآورد می‌شود. <a href="#fnref-free-location" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-provenance" dir="rtl" align="right"><strong>۱۵.</strong> <bdi dir="ltr">Provenance</bdi> — اطلاعات ثبت‌شده دربارهٔ نسخهٔ منبع، محیط و اجرا تا یک ادعا بعداً قابل بررسی باشد. <a href="#fnref-provenance" aria-label="بازگشت به متن">↩</a></p>
