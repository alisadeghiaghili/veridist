<a id="veridist-api-fa"></a>
<h1 dir="rtl" align="right">راهنمای API در <bdi dir="ltr">Veridist</bdi></h1>

<p dir="rtl" align="right">این راهنما API عمومیِ فعلی <bdi dir="ltr">Veridist</bdi> را توضیح می‌دهد. برای برازش توزیع<sup id="fnref-fitting"><a href="#fn-fitting">۱</a></sup> نمایی روی داده‌های طول عمر، معمولاً یک فایل CSV را به تابع اصلی می‌دهید و نتیجه را دریافت می‌کنید. اگر محاسبه طولانی است و ممکن است متوقف شود، می‌توانید پیشرفت آن را ذخیره کنید و بعداً ادامه دهید. ابزارهای اسکالر<sup id="fnref-scalar"><a href="#fn-scalar">۲</a></sup> و جریان‌های داده‌ای که مدیریتشان با فراخواننده<sup id="fnref-caller"><a href="#fn-caller">۳</a></sup> است نیز برای استفاده‌های فنی‌تر در دسترس‌اند.</p>

<h2 dir="rtl" align="right">مسیر اجرا را انتخاب کنید</h2>

<table dir="rtl" width="100%">
  <thead>
    <tr>
      <th width="36%" align="center"><p align="center">کاری که می‌خواهید انجام دهید</p></th>
      <th width="32%" align="center"><p align="center">تابع مربوط</p></th>
      <th width="32%" align="center"><p align="center">محدودیت این روش</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right">خواندن فایل CSV و برازش مدل نمایی در یک اجرا</td>
      <td dir="ltr" align="left"><code>fit_exponential_csv</code></td>
      <td align="right">امکان توقف و ادامه، لغو اجرا یا انتخاب خودکار مدل را ندارد.</td>
    </tr>
    <tr>
      <td align="right">ادامهٔ محاسبه با داده‌های JSON بخش‌بندی‌شده</td>
      <td dir="ltr" align="left"><code>fit_exponential_checkpointed_chunks</code></td>
      <td align="right">آماده‌کردن و خواندن بخش‌های داده با برنامهٔ شماست.</td>
    </tr>
    <tr>
      <td align="right">پردازش فایل CSV با امکان لغو و ادامه از آخرین سطر ذخیره‌شده</td>
      <td dir="ltr" align="left"><code>fit_exponential_checkpointed_csv</code></td>
      <td align="right">فقط برای فایل محلی روی یک رایانه است؛ بازیابی توزیع‌شده<sup id="fnref-distributed-recovery"><a href="#fn-distributed-recovery">۴</a></sup> ندارد.</td>
    </tr>
    <tr>
      <td align="right">محاسبهٔ چگالی لگاریتمی برای یک مقدار</td>
      <td dir="ltr" align="left"><code>evaluate_log_density</code></td>
      <td align="right">پارامترها را برازش نمی‌کند.</td>
    </tr>
    <tr>
      <td align="right">محاسبهٔ درست‌نمایی برای چند بخش داده</td>
      <td dir="ltr" align="left"><code>reduce_log_likelihood_chunks</code></td>
      <td align="right">بهترین توزیع را انتخاب نمی‌کند.</td>
    </tr>
  </tbody>
</table>

<p dir="rtl" align="right">برای حالت معمول، نمونهٔ کامل زیر را اجرا کنید:</p>

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

<p dir="rtl" align="right">در این مثال دو مشاهده داریم. در مشاهدهٔ اول، رویداد در زمان ۱ رخ داده است. در مشاهدهٔ دوم، تا زمان ۱ رویدادی ندیده‌ایم و فقط می‌دانیم عمر واقعی از ۱ بیشتر است. بنابراین یک رخداد ثبت‌شده و مجموعاً ۲ واحد زمانِ تحت مشاهده داریم. نرخ<sup id="fnref-rate"><a href="#fn-rate">۵</a></sup> برآوردشده برابر <code dir="ltr">1 / 2 = 0.5</code> رخداد در هر واحد زمان است. این عدد احتمال ۵۰ درصدی رخداد نیست؛ واحد آن به واحد ستون <code dir="ltr">time</code> وابسته است.</p>

<h2 dir="rtl" align="right">فایل CSV باید چه شکلی باشد؟</h2>

<p dir="rtl" align="right">تابع <code dir="ltr">fit_exponential_csv(path, *, schema, source_id, limits)</code> یک فایل UTF-8 با دو ستون <code dir="ltr">time,event_observed</code> و دقیقاً همین ترتیب می‌پذیرد. مقدار <code dir="ltr">time</code> باید عددی متناهی و نامنفی باشد. در ستون <code dir="ltr">event_observed</code> مقدار <code dir="ltr">1</code> یعنی رویداد در زمان ثبت‌شده رخ داده است؛ مقدار <code dir="ltr">0</code> یعنی تا پایان زمان ثبت‌شده هنوز رویداد را ندیده‌ایم. حالت دوم سانسور راست مستقل<sup id="fnref-right-censoring"><a href="#fn-right-censoring">۶</a></sup> نام دارد.</p>

<p dir="rtl" align="right"><code dir="ltr">CsvLifetimeSchema</code> نام دو ستون مورد انتظار را مشخص می‌کند. <code dir="ltr">PublicSourceId</code> یک شناسهٔ عمومی و غیرمحرمانه برای ثبت منشأ<sup id="fnref-provenance"><a href="#fn-provenance">۷</a></sup> داده است؛ مسیر محلی فایل در گزارش نتیجه قرار نمی‌گیرد. <code dir="ltr">CsvLifetimeLimits</code> سقف اندازهٔ هر بخش از داده و حداکثر حجم داده‌ای را که هم‌زمان در صف پردازش نگه داشته می‌شود تعیین می‌کند<sup id="fnref-byte-limits"><a href="#fn-byte-limits">۸</a></sup>. هر دو مقدار باید مثبت باشند.</p>

<p dir="rtl" align="right"><bdi dir="ltr">Veridist</bdi> نام ستون‌ها، جداکننده، کدگذاری، دادهٔ گمشده یا معنی صفر و یک را حدس نمی‌زند. اگر فایل با قرارداد بالا سازگار نباشد، مشکل را صریح گزارش می‌کند و داده را پنهانی تغییر نمی‌دهد.</p>

<h2 dir="rtl" align="right">نتیجه را چگونه بخوانید؟</h2>

<p dir="rtl" align="right">تابع <code dir="ltr">fit_exponential_csv</code> همیشه یک شیء از نوع <code dir="ltr">ExponentialSourceFitResult</code> برمی‌گرداند. اگر فایل با موفقیت پردازش شود و داده برای برآورد نرخ کافی باشد، نتیجهٔ برازش در <code dir="ltr">result.fit</code> قرار می‌گیرد. اگر فایل درست پردازش شود ولی از نظر آماری نتوان نرخ معتبری برآورد کرد—برای مثال فایل خالی باشد یا هیچ رخدادی مشاهده نشده باشد—همان فیلد یک عدم‌برآورد آماری نوع‌دار<sup id="fnref-typed-non-estimate"><a href="#fn-typed-non-estimate">۹</a></sup> برمی‌گرداند تا دلیل مشخص بماند.</p>

<p dir="rtl" align="right">اگر خواندن یا پردازش فایل شکست بخورد، <code dir="ltr">result.fit</code> برابر <code dir="ltr">None</code> است. در این حالت <code dir="ltr">result.execution</code> یک خروجی نوع‌دار<sup id="fnref-typed-outcome"><a href="#fn-typed-outcome">۱۰</a></sup> شامل مرحله و دلیل شکست دارد. پیش از استفاده از پارامترهای مدل، ابتدا نوع نتیجه را بررسی کنید.</p>

<p dir="rtl" align="right">مدل فعلی پارامتر مکان را روی صفر ثابت نگه می‌دارد. این مسیر هنوز فاصلهٔ اطمینان<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">۱۱</a></sup>، آزمون مناسب‌بودن مدل<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">۱۲</a></sup>، وزن<sup id="fnref-weight"><a href="#fn-weight">۱۳</a></sup>، متغیر کمکی<sup id="fnref-covariate"><a href="#fn-covariate">۱۴</a></sup>، برش داده<sup id="fnref-truncation"><a href="#fn-truncation">۱۵</a></sup>، سانسور چپ<sup id="fnref-left-censoring"><a href="#fn-left-censoring">۱۶</a></sup>، سانسور فاصله‌ای<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">۱۷</a></sup>، پارامتر مکان آزاد<sup id="fnref-free-location"><a href="#fn-free-location">۱۸</a></sup> یا انتخاب خودکار مدل را ارائه نمی‌کند. جزئیات فرض آماری و حالت‌های ناموفق در <a href="exponential-right-censoring.md">آموزش سانسور راست</a> آمده است.</p>

<h2 dir="rtl" align="right">ذخیرهٔ پیشرفت و ادامهٔ محاسبه</h2>

<p dir="rtl" align="right">برای داده‌ای که برنامهٔ شما از قبل به بخش‌های کوچک JSON تقسیم کرده است، تابع <code dir="ltr">fit_exponential_checkpointed_chunks</code> را از پکیج <code dir="ltr">veridist</code> وارد کنید. این تابع هر بخش را جداگانه پردازش می‌کند و آمار کافی<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">۱۹</a></sup> لازم برای ادامهٔ محاسبه را ذخیره می‌کند؛ نسخه‌ای از سطرهای اصلی داده را در فایل وضعیت نگه نمی‌دارد.</p>

<p dir="rtl" align="right">برای فایل CSV، تابع <code dir="ltr">veridist.execution.fit_exponential_checkpointed_csv</code> همین کار را همراه با امکان لغو انجام می‌دهد. شما فایل، تعریف ستون‌ها، محدودیت اندازه، محل ذخیرهٔ وضعیت، شناسهٔ نسخهٔ داده و در صورت نیاز تابع <code dir="ltr">cancel(cursor)</code> را می‌دهید. اگر اجرا لغو شود، بخش کامل‌شده ابتدا ذخیره می‌شود؛ اجرای بعدی می‌تواند از آخرین سطر ثبت‌شده ادامه دهد.</p>

<p dir="rtl" align="right">فایل داده و شناسهٔ نسخهٔ آن باید بین دو اجرا ثابت بمانند. سازوکار ذخیرهٔ وضعیت<sup id="fnref-checkpoint"><a href="#fn-checkpoint">۲۰</a></sup> برای ادامهٔ محاسبه روی همان رایانه طراحی شده است. پیاده‌سازی فعلی از SQLite محلی استفاده می‌کند، اما سطرهای خام CSV را در آن کپی نمی‌کند. این فایل به‌صورت خودکار رمزگذاری یا احراز هویت نمی‌شود؛ بنابراین باید آن را مانند سایر فایل‌های کاری در محل امن نگه دارید. <a href="../../examples/checkpoint_resume.py">مثال ذخیره و ادامهٔ اجرا</a> روند دو مرحله‌ای را نشان می‌دهد.</p>

<h2 dir="rtl" align="right">ابزارهای سطح پایین<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">۲۱</a></sup> برای داده‌های آماده</h2>

<p dir="rtl" align="right">اگر داده را خود برنامهٔ شما تولید یا بخش‌بندی می‌کند، <code dir="ltr">IterableDataSource</code> آن بخش‌ها را همراه با مشخصات منبع دریافت می‌کند. در حالت <code dir="ltr">SINGLE_PASS</code> داده فقط یک‌بار خوانده می‌شود. در حالت <code dir="ltr">REPLAYABLE</code> باید تابعی بدهید که هر بار یک پیمایش تازه از داده بسازد. حالت <code dir="ltr">CHECKPOINT_REPLAYABLE</code> در این adapter هنوز پیاده نشده است؛ استفاده از آن خطای <code dir="ltr">CHECKPOINT_REQUIRED</code> می‌دهد.</p>

<p dir="rtl" align="right"><code dir="ltr">FAMILY_REGISTRY</code> و <code dir="ltr">FamilyId</code> مشخصات پنج خانوادهٔ آماری ارزیابی‌شده را نگه می‌دارند. <code dir="ltr">evaluate_log_density</code> چگالی لگاریتمی یک مقدار را با پارامترهای داده‌شده محاسبه می‌کند. <code dir="ltr">reduce_log_likelihood_chunks</code> همین محاسبه را برای بخش‌های متعدد داده جمع می‌کند و نتیجه‌ای مستقل از نحوهٔ بخش‌بندی می‌سازد. این توابع پارامترهای مدل را تخمین نمی‌زنند، توزیع‌ها را رتبه‌بندی نمی‌کنند و برای دادهٔ سانسورشده درست‌نمایی نمی‌سازند. قرارداد دقیق آن‌ها در <a href="families-log-density-likelihood.md">راهنمای خانواده‌ها و درست‌نمایی لگاریتمی</a> آمده است.</p>

<details dir="rtl" align="right">
<summary>جزئیات فنی و محدودیت‌های نسخهٔ فعلی</summary>

<p dir="rtl" align="right">خواندن فایل CSV در یک پیمایش انجام می‌شود. محدودیت‌های اندازه فقط حجم بخش‌های داده‌ای را که خود adapter نگه می‌دارد کنترل می‌کنند و سقف کل حافظهٔ فرایند یا سرعت اجرا نیستند. پشتیبانی عمومی از همهٔ منابع داده، اجرای توزیع‌شده و بازیابی بین چند رایانه در نسخهٔ فعلی وجود ندارد؛ موارد برنامه‌ریزی‌شده و مرزهای دقیق در <a href="../../KNOWN_LIMITS.fa.md">محدودیت‌های شناخته‌شده</a> ثبت شده‌اند. موفق‌بودن محاسبه نیز به‌تنهایی ثابت نمی‌کند که توزیع نمایی برای داده مناسب است.</p>

</details>

<h2 dir="rtl" align="right">اصطلاحات این صفحه</h2>

<p id="fn-fitting" dir="rtl" align="right"><strong>۱.</strong> <bdi dir="ltr">Distribution fitting</bdi> — برآورد پارامترهای یک توزیع از روی داده و بررسی سازگاری آن با مشاهدات. <a href="#fnref-fitting" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-scalar" dir="rtl" align="right"><strong>۲.</strong> <bdi dir="ltr">Scalar operation</bdi> — عملیاتی که هر بار روی یک مقدار عددی کار می‌کند، نه روی یک آرایهٔ کامل. <a href="#fnref-scalar" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-caller" dir="rtl" align="right"><strong>۳.</strong> <bdi dir="ltr">Caller</bdi> — کد یا برنامه‌ای که تابع کتابخانه را صدا می‌زند و ورودی‌هایش را فراهم می‌کند. <a href="#fnref-caller" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-distributed-recovery" dir="rtl" align="right"><strong>۴.</strong> <bdi dir="ltr">Distributed recovery</bdi> — ادامه‌دادن یک محاسبه روی رایانه یا سرویس دیگری با وضعیت مشترک. <a href="#fnref-distributed-recovery" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-rate" dir="rtl" align="right"><strong>۵.</strong> <bdi dir="ltr">Rate</bdi> — تعداد مورد انتظار رخداد در هر واحد زمان؛ نرخ با احتمال رخداد یکسان نیست. <a href="#fnref-rate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-right-censoring" dir="rtl" align="right"><strong>۶.</strong> <bdi dir="ltr">Independent right censoring</bdi> — تا پایان مشاهده رخداد دیده نشده و فرض می‌شود علت پایان مشاهده از زمان رخداد مستقل است. <a href="#fnref-right-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-provenance" dir="rtl" align="right"><strong>۷.</strong> <bdi dir="ltr">Provenance</bdi> — اطلاعات ثبت‌شده دربارهٔ منبع داده و نحوهٔ اجرای محاسبه برای بررسی بعدی نتیجه. <a href="#fnref-provenance" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-byte-limits" dir="rtl" align="right"><strong>۸.</strong> <bdi dir="ltr">Byte limits</bdi> — سقف حجم داده‌ای که هر بخش و صف پردازش می‌توانند هم‌زمان نگه دارند؛ این مقدار سقف کل RAM برنامه نیست. <a href="#fnref-byte-limits" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-typed-non-estimate" dir="rtl" align="right"><strong>۹.</strong> <bdi dir="ltr">Typed statistical non-estimate</bdi> — نتیجه‌ای ساخت‌یافته که می‌گوید محاسبه اجرا شده اما داده اجازهٔ برآورد معتبر را نداده است و دلیل را مشخص می‌کند. <a href="#fnref-typed-non-estimate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-typed-outcome" dir="rtl" align="right"><strong>۱۰.</strong> <bdi dir="ltr">Typed outcome</bdi> — نتیجه‌ای با نوع و کد مشخص که برنامه می‌تواند بدون تحلیل متن آزاد آن را بررسی کند. <a href="#fnref-typed-outcome" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-confidence-interval" dir="rtl" align="right"><strong>۱۱.</strong> <bdi dir="ltr">Confidence interval</bdi> — بازه‌ای که عدم‌قطعیت برآورد پارامتر را با یک روش آماری مشخص نشان می‌دهد. <a href="#fnref-confidence-interval" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-goodness-of-fit" dir="rtl" align="right"><strong>۱۲.</strong> <bdi dir="ltr">Goodness-of-fit test</bdi> — بررسی آماری میزان سازگاری شکل توزیع انتخاب‌شده با داده. <a href="#fnref-goodness-of-fit" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-weight" dir="rtl" align="right"><strong>۱۳.</strong> <bdi dir="ltr">Weight</bdi> — عددی که سهم یک مشاهده را در محاسبه بیشتر یا کمتر می‌کند. <a href="#fnref-weight" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-covariate" dir="rtl" align="right"><strong>۱۴.</strong> <bdi dir="ltr">Covariate</bdi> — متغیری مانند دما یا فشار که ممکن است روی زمان رخداد اثر بگذارد. <a href="#fnref-covariate" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-truncation" dir="rtl" align="right"><strong>۱۵.</strong> <bdi dir="ltr">Truncation</bdi> — حذف‌شدن بخشی از جامعه از نمونه به‌علت سازوکار ورود یا مشاهده، نه صرفاً نامشخص‌بودن زمان رخداد. <a href="#fnref-truncation" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-left-censoring" dir="rtl" align="right"><strong>۱۶.</strong> <bdi dir="ltr">Left censoring</bdi> — فقط می‌دانیم رخداد پیش از یک زمان مشخص اتفاق افتاده است. <a href="#fnref-left-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-interval-censoring" dir="rtl" align="right"><strong>۱۷.</strong> <bdi dir="ltr">Interval censoring</bdi> — فقط می‌دانیم رخداد در فاصلهٔ بین دو زمان اتفاق افتاده است. <a href="#fnref-interval-censoring" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-free-location" dir="rtl" align="right"><strong>۱۸.</strong> <bdi dir="ltr">Free location parameter</bdi> — پارامتر جابه‌جایی توزیع که به‌جای ثابت‌بودن، از داده برآورد می‌شود. <a href="#fnref-free-location" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-sufficient-statistics" dir="rtl" align="right"><strong>۱۹.</strong> <bdi dir="ltr">Sufficient statistics</bdi> — خلاصه‌های عددی لازم برای برآورد پارامتر که در این محاسبه جای نگهداری همهٔ سطرهای خام را می‌گیرند. <a href="#fnref-sufficient-statistics" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-checkpoint" dir="rtl" align="right"><strong>۲۰.</strong> <bdi dir="ltr">Checkpoint</bdi> — وضعیت میانی ذخیره‌شده که اجازه می‌دهد اجرای سازگار از آخرین بخش ثبت‌شده ادامه پیدا کند. <a href="#fnref-checkpoint" aria-label="بازگشت به متن">↩</a></p>
<p id="fn-low-level-tools" dir="rtl" align="right"><strong>۲۱.</strong> <bdi dir="ltr">Low-level tools</bdi> — توابع پایه‌ای‌تر برای زمانی که برنامهٔ شما آماده‌سازی داده، بخش‌بندی داده یا انتخاب پارامترها را خودش انجام می‌دهد. این ابزارها معمولاً برای مسیر شروع سریع کاربر نهایی نیستند. <a href="#fnref-low-level-tools" aria-label="بازگشت به متن">↩</a></p>
